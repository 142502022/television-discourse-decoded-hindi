# Pending Tasks

This file tracks the remaining work for the English news overlap-annotation pipeline.

## Current Status

Some engineering pieces already exist in code:

- Download videos from YouTube IDs or a channel URL.
- Extract audio from downloaded videos.
- Generate 480p H.264 proxy videos.
- Run WhisperX for transcript, word timing, and anonymous speaker diarization.
- Run Pyannote OSD for overlapped speech intervals.
- Merge WhisperX diarization intersections with Pyannote OSD intervals.
- Build per-episode JSON files.
- Chunk proxy videos into 15-minute pieces.
- Export Label Studio tasks, one task per chunk.

## Pending Work

- Confirm that all 5 selected episodes download successfully.
- Confirm audio extraction succeeds for every downloaded episode.
- Confirm 480p proxy generation succeeds for every downloaded episode.
- Fix any remaining runtime environment issues for WhisperX, Pyannote, Torch, TorchVision, or CUDA.
- Run WhisperX successfully on all selected episodes.
- Run Pyannote OSD successfully on all selected episodes.
- Review overlap candidate intervals from:
  - both WhisperX diarization intersections and Pyannote OSD agreeing,
  - Pyannote OSD only,
  - diarization intersection only.
- Decide the confidence rules for overlap candidates after inspecting real outputs.
- Implement or run LLM-based roster extraction for participants.
- Extract participant names, roles, affiliations, gender-as-addressed, and evidence timestamps.
- Randomly sample 10 evidence timestamps across all participants and videos.
- Manually verify sampled evidence timestamps against the proxy videos.
- Record evidence verification accuracy and error statistics.
- Link anonymous speaker clusters, such as `SPEAKER_01`, to verified participant names.
- Finalize the per-episode JSON schema after sanity checks.
- Decide the exact Label Studio annotation interface.
- Run 15-minute chunking on finalized proxy videos.
- Slice finalized episode JSON with respect to each chunk.
- Import generated chunk tasks into Label Studio.

## Honest Limitations

- Roster extraction is not complete in code yet. Placeholder fields exist, but participant identities are not generated or verified.
- Speaker-cluster-to-name linking is not complete yet.
- Manual evidence verification cannot be automated fully; it requires human review.
- Label Studio export code exists, but the final task format may need adjustment after the final JSON schema is accepted.
