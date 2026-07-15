from youtube_transcript_api import YouTubeTranscriptApi

video_ids = [
    "JFF2vJaN0Cw",
    "4D55Cmj2t-A",
    "n0iaPtsnmxQ",
    "GfaHdjApnhU",
    "lg-f92uY1Lc",
    "uDulBxDb7GM",
    "7t0YJWTjmdI",
    "3IaB2a8tXLA",
    "YxyLN3N5w9s",
    "wuI6FGsOFZU",
    "mf4bRP_puNQ",
    "3N5a9cHYzCM",
    "dDP36_ZBs6A",
    "vdtqEPKYB5M",
    "JhBnOamc_8s"
]

all_transcripts = ""

for vid in video_ids:
    print(f"Fetching transcript for {vid}...")
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=['en', 'en-IN'])
        transcript_text = " ".join([t['text'] for t in transcript_list])
        all_transcripts += f"\n\n--- VIDEO ID: {vid} ---\n{transcript_text}"
    except Exception as e:
        print(f"Error fetching {vid}: {e}")
        all_transcripts += f"\n\n--- VIDEO ID: {vid} ---\n[TRANSCRIPT UNAVAILABLE: {e}]"

with open("all_transcripts.txt", "w") as f:
    f.write(all_transcripts)

print("Saved to all_transcripts.txt")
