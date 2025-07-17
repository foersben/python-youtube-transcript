from typing import Any
from src.utils import file_io, json_utils
from src.core.gemini import request_gemini
import logging

# Configure logging
logger = logging.getLogger(__name__)


def create_section_timestamps(
    transcript: list[dict[str, Any]],
    section_count_range: tuple[int, int] = (15, 20),
    title_length_range: tuple[int, int] = (3, 8),
    output_file: str | None = None,
) -> list[dict[str, Any]]:
    """Generates YouTube-style section timestamps with optional speaker info.

    Args:
        transcript: Transcript data from extract_transcript()
        section_count_range: Number of sections [min, max]
        title_length_range: Words in title [min, max]
        output_file: Optional file path to save sections

    Returns:
        List of sections with start times and titles
    """

    try:
        # Format transcript with speaker info if available
        transcript_lines = []
        for seg in transcript:
            speaker = f"[{seg['speaker']}] " if "speaker" in seg else ""
            transcript_lines.append(f"[{seg['start']:.1f}s] {speaker}{seg['text']}")
        formatted_transcript = "\n".join(transcript_lines)

        prompt = f"""
        Create YouTube-style chapter markers from this transcript.
        Return ONLY valid JSON with this structure:
        [{"start": seconds, 'title': string}, ...]

        Rules:
        1. Create {section_count_range[0]}-{section_count_range[1]} sections
        2. Start time in seconds (float)
        3. {title_length_range[0]}-{title_length_range[1]} word titles
        4. Capture key topics
        5. Output ONLY the JSON array
        6. In the language of the script

        Transcript:
        {formatted_transcript}
        """

        response_text = request_gemini(prompt)
        sections = json_utils.extract_json(response_text)

        # Validate sections
        if not isinstance(sections, list) or not all(
            isinstance(sec, dict) and all(k in sec for k in ["start", "title"])
            for sec in sections
        ):
            raise ValueError("Invalid section format in AI response")

        # Save output
        if output_file:
            file_io.write_to_file(sections, output_file)
            logger.info(f"Section timestamps saved to {output_file}")

        return sections

    except Exception as e:
        logger.error(f"Error generating sections: {e}")
        raise
