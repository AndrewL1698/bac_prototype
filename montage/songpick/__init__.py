"""Phase 12: pick a song for an album from the user's Spotify listening history.

    from montage.songpick import load_history, shortlist
    pick = shortlist(load_history([history_dir]), [a.taken_at for a in assets])

The app only shortlists; the user supplies the audio file (see docs/spec/12_songpick.md).
"""

from montage.songpick.history import load_history
from montage.songpick.pick import shortlist

__all__ = ["load_history", "shortlist"]
