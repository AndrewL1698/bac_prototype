"""Phase 12: pick a song for an album from the user's Spotify listening history.

    from montage.songpick import suggest, spotify
    recent, top = spotify.fetch_listening(token)            # after "Connect Spotify"
    pick = suggest([a.taken_at for a in assets], recent=recent, top=top)
    # optional: export=load_history([history_dir]) for photos older than about a year

Audio: montage.songpick.youtube (opt-in) or the user's own upload (see docs/spec/12_songpick.md).
"""

from montage.songpick.history import load_history
from montage.songpick.pick import shortlist, suggest

__all__ = ["load_history", "shortlist", "suggest"]
