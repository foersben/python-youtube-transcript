from typing import Any, Optional, List, Dict
import re
import os
from src.utils import file_io
from youtube_transcript_api import YouTubeTranscriptApi
from src.core.gemini import request_gemini
from pytube import YouTube
from pydub import AudioSegment
import logging
import json

# Configure logging
logger = logging.getLogger(__name__)


def _speaker_recognition(
    transcript: List[Dict[str, Any]],
    audio_file_path: str,
    known_speakers: Optional[List[Dict[str, str]]] = None,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Enhances transcript with speaker recognition using audio analysis.

    Args:
        transcript: List of transcript segments with keys:
            - 'text': Segment text
            - 'start': Start time in seconds
            - 'duration': Duration in seconds
        audio_file_path: Path to audio file for speaker diarization
        known_speakers: Optional list of known speakers with descriptions
            Format: [{"name": "John", "description": "Deep voice"}, ...]
        output_file: Optional file path to save enhanced transcript

    Returns:
        Enhanced transcript with 'speaker' field added to each segment

    Raises:
        RuntimeError: If speaker recognition fails
    """

    try:
        # Step 1: Enhance transcript with speaker change detection
        transcript_str = "\n".join(
            f"[{seg['start']:.1f}s - {seg['start'] + seg['duration']:.1f}s]: {seg['text']}"
            for seg in transcript
        )

        prompt = f"""
        Analyze the following transcript and identify speaker changes based on content:
        {transcript_str}
        
        Return ONLY valid JSON with the same structure as the input transcript but with split segments so that the segment text is only spoken by one person. 
        [{"text": string, 'start': seconds, 'duration': seconds}, ...]

        Rules:
        1. If a segment has multiple speakers, split it into multiple entries
        2. Segments resulting from a split segment have in the sum the same duration
        3. Exactly one segment of the segments resulting from the split starts at the original start time.
        4. The other start times are in between the original start time and the start time + the original segments duration.
        5. Output ONLY the JSON array
        """

        enhanced_transcript = request_gemini(
            prompt=prompt,
            filepaths=[audio_file_path] if os.path.exists(audio_file_path) else [],
        )

        speaker_context = (
            "No known speakers provided"
            if not known_speakers
            else (
                "Known speakers:\n"
                + "\n".join(
                    f"- {sp['name']}: {sp['description']}" for sp in known_speakers
                )
            )
        )

        # Step 2: Perform speaker diarization with audio
        prompt = f"""
        Using the audio file, perform speaker diarization on this transcript:
        {json.dumps(enhanced_transcript)}
        
        Known speakers:
        {speaker_context}
        
        Return ONLY valid JSON with the same structure but with updated 'speaker' field.
        [{"text": string, 'start': seconds, 'duration': seconds, 'speaker': name}, ...]

        Use these rules:
        
        1. Assign speaker names based on voice characteristics
        2. For unknown speakers, use "UnknownX" where X is an incrementing number for each new unknown speaker
        3. Output ONLY the JSON array
        """

        final_transcript = request_gemini(prompt=prompt, filepaths=[audio_file_path])

        # Save if requested
        if output_file:
            file_io.write_to_file(final_transcript, output_file)
            logger.info(f"Enhanced transcript saved to {output_file}")

        return final_transcript

    except Exception as e:
        logger.error(f"Speaker recognition failed: {e}")
        raise RuntimeError(f"Speaker recognition failed: {e}")


def _extract_audio(
    video_id: str, output_dir: str = "./", filename: str = "audio"
) -> str:
    """Extracts audio from YouTube video and converts to MP3.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save audio file
        filename: Base filename (without extension)

    Returns:
        Path to the extracted MP3 file

    Raises:
        RuntimeError: If audio extraction fails
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Download audio
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        audio_stream = yt.streams.filter(only_audio=True).first()

        if not audio_stream:
            raise ValueError("No audio stream available")

        # Download and convert to MP3
        temp_file = audio_stream.download(
            output_path=output_dir, filename=f"{filename}_temp"
        )
        audio = AudioSegment.from_file(temp_file)
        mp3_path = os.path.join(output_dir, f"{filename}.mp3")
        audio.export(mp3_path, format="mp3")

        # Cleanup temporary file
        os.remove(temp_file)

        logger.info(f"Audio extracted and saved to {mp3_path}")
        return mp3_path

    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        raise RuntimeError(f"Audio extraction failed: {e}")


def _convert_transcript_to_dict(transcript_data) -> List[Dict[str, Any]]:
    """Converts transcript data to a serializable list of dictionaries.

    Args:
        transcript_data: Raw transcript data from YouTubeTranscriptApi

    Returns:
        List of dictionaries with:
        - 'text': Transcript text
        - 'start': Start time in seconds
        - 'duration': Duration in seconds
    """

    return [
        {"text": segment.text, "start": segment.start, "duration": segment.duration}
        for segment in transcript_data
    ]


def extract_transcript(
    video_id: str,
    output_file: Optional[str] = None,
    translate_to: Optional[str] = None,
    enhance_with_speakers: bool = False,
    known_speakers: Optional[List[Dict[str, str]]] = None,
    audio_output_dir: str = "./audio",
) -> List[Dict[str, Any]]:
    """Extracts YouTube transcript and optionally enhances it with speaker recognition.

    Args:
        video_id: YouTube video ID (11-character string)
        output_file: Optional file path to save transcript
        translate_to: Optional language code for translation (e.g., 'en')
        enhance_with_speakers: Whether to perform speaker recognition
        known_speakers: Optional list of known speakers for recognition
            Format: [{"name": "John", "description": "Deep voice"}, ...]
        audio_output_dir: Directory to store extracted audio files

    Returns:
        List of transcript segments as dictionaries, optionally with 'speaker' field

    Raises:
        Exception: If transcript retrieval fails
    """

    try:
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcripts = list(transcript_list)

        # Find the first manually created transcript, or fallback to generated
        transcript = next(
            (t for t in transcripts if not t.is_generated),
            transcripts[0],
        )

        # Fetch and optionally translate
        if translate_to:
            transcript_data = transcript.translate(translate_to).fetch()
        else:
            transcript_data = transcript.fetch()

        serializable_data = _convert_transcript_to_dict(transcript_data)

        # Print transcript metadata
        logger.info(f"\nTranscript Details:")
        logger.info(f"- Video ID: {transcript.video_id}")
        logger.info(f"- Language: {transcript.language} ({transcript.language_code})")
        logger.info(f"- Generated: {'Yes' if transcript.is_generated else 'No'}")

        # Enhance with speaker recognition if requested
        if enhance_with_speakers:
            audio_path = _extract_audio(
                video_id, output_dir=audio_output_dir, filename=f"audio_{video_id}"
            )
            serializable_data = _speaker_recognition(
                transcript=serializable_data,
                audio_file_path=audio_path,
                known_speakers=known_speakers,
            )

        # Save to file if requested
        if output_file:
            file_io.write_to_file(serializable_data, output_file)
            logger.info(f"Transcript saved to {output_file}")

        return serializable_data

    except Exception as e:
        logger.error(f"Error retrieving transcript: {e}")
        raise


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL

    Args:
        url: ID or YouTube URL from which video ID has to be extracted

    Returns:
        Video ID

    Raises:
        ValueError: If invalid YouTube URL or ID is passed
    """

    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL or ID")
