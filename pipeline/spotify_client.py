"""
Spotify Client
===============
Handles Spotify API authentication (Client Credentials flow) and
track search to retrieve Spotify track IDs for embedding the
Spotify Web Player in the Streamlit UI.

Uses Client Credentials flow — no user login required.
Only requires CLIENT_ID and CLIENT_SECRET.
"""

import os
import time
from typing import Dict, Optional

import requests


SPOTIFY_CLIENT_ID = "20ff8ab376b248f19c4e9ba4aa076ef8"
SPOTIFY_CLIENT_SECRET = "3dedbe92e7f84bfcb9a473e5dc43a2cf"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"


class SpotifyClient:
    """
    Lightweight Spotify API client using Client Credentials flow.

    Authenticates automatically on first use and refreshes the token
    when it expires. Provides track search by name + artist and returns
    the Spotify track ID needed to build the embed URL.

    Attributes:
        client_id (str): Spotify app client ID.
        client_secret (str): Spotify app client secret.
        _access_token (str | None): Cached access token.
        _token_expiry (float): Unix timestamp when the token expires.
    """

    def __init__(
        self,
        client_id: str = SPOTIFY_CLIENT_ID,
        client_secret: str = SPOTIFY_CLIENT_SECRET,
    ) -> None:
        """
        Initialize the SpotifyClient.

        Args:
            client_id: Spotify application client ID.
            client_secret: Spotify application client secret.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    def _authenticate(self) -> bool:
        """
        Fetch a new access token using Client Credentials flow.

        Returns:
            True if authentication succeeded, False otherwise.
        """
        try:
            response = requests.post(
                SPOTIFY_TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                self._access_token = data["access_token"]
                expires_in = data.get("expires_in", 3600)
                self._token_expiry = time.time() + expires_in - 60
                return True
            else:
                print(
                    f"[SpotifyClient] Auth failed: {response.status_code} "
                    f"{response.text[:200]}"
                )
                return False
        except Exception as exc:
            print(f"[SpotifyClient] Auth error: {exc}")
            return False

    def _get_token(self) -> Optional[str]:
        """
        Return a valid access token, refreshing if expired.

        Returns:
            Valid access token string, or None if auth failed.
        """
        if self._access_token is None or time.time() >= self._token_expiry:
            if not self._authenticate():
                return None
        return self._access_token

    @staticmethod
    def _is_match(expected_track: str, expected_artist: str, result_track: str, result_artist: str) -> bool:
        """
        Validate that a Spotify result actually matches the expected track and artist.

        Uses case-insensitive partial matching — the expected name must appear
        somewhere within the returned name (or vice-versa).

        Args:
            expected_track: The track name we searched for.
            expected_artist: The artist name we searched for.
            result_track: The track name returned by Spotify.
            result_artist: The artist name returned by Spotify.

        Returns:
            True if both track and artist are considered a match, False otherwise.
        """
        et = expected_track.lower().strip()
        ea = expected_artist.lower().strip()
        rt = result_track.lower().strip()
        ra = result_artist.lower().strip()

        track_match = et in rt or rt in et
        artist_match = ea in ra or ra in ea
        return track_match and artist_match

    def search_track(
        self, track_name: str, artist_name: str
    ) -> Optional[Dict]:
        """
        Search Spotify for a track by name and artist.

        Uses the strict ``track:"..." artist:"..."`` query and checks up to 5
        results for a match.  Returns the first result whose track name and
        artist name pass case-insensitive partial-match validation, or ``None``
        if no valid match is found.

        Args:
            track_name: Name of the track to search for.
            artist_name: Name of the artist.

        Returns:
            Dict with keys: track_id, track_name, artist_name, album_art_url, embed_url.
            Returns None if no match found or request fails.
        """
        token = self._get_token()
        if token is None:
            return None

        query = f'track:"{track_name}" artist:"{artist_name}"'

        try:
            response = requests.get(
                SPOTIFY_SEARCH_URL,
                headers={"Authorization": f"Bearer {token}"},
                params={"q": query, "type": "track", "limit": 5},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                items = data.get("tracks", {}).get("items", [])
                for item in items:
                    result_track = item.get("name", "")
                    result_artist = item["artists"][0]["name"] if item.get("artists") else ""
                    if self._is_match(track_name, artist_name, result_track, result_artist):
                        track_id = item["id"]
                        images = item.get("album", {}).get("images", [])
                        album_art = images[0]["url"] if images else None
                        return {
                            "track_id": track_id,
                            "track_name": result_track,
                            "artist_name": result_artist,
                            "album_art_url": album_art,
                            "embed_url": f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0",
                            "spotify_url": item.get("external_urls", {}).get("spotify", ""),
                        }
                # No result passed validation
                print(
                    f"[SpotifyClient] No matching result for '{track_name}' by '{artist_name}' "
                    f"among {len(items)} candidate(s)."
                )
            else:
                print(
                    f"[SpotifyClient] Search failed: {response.status_code} "
                    f"{response.text[:200]}"
                )
        except Exception as exc:
            print(f"[SpotifyClient] Search error for '{track_name}': {exc}")

        return None

