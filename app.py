"""
動画→説明記事 変換ツール（クラウド版）
ブラウザで動画をアップロードするだけで、画像つきの説明記事を自動生成します。
完全無料：Gemini API（文字起こし＋記事生成）
"""

import json
import os
import re
import subprocess
import tempfile
import shutil
import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# --- ページ設定 ---
st.set_page_config(
    page_title="動画→記事 変換ツール",
    page_icon="🎬",
    layout="centered",
)

# --- スタイル ---
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 定数 ---
SCRIPT_DIR = Path(__file__).resolve().parent
GUIDE_PATH = SCRIPT_DIR / "つかいかた.md"


def check_ffmpeg():
    """FFmpegが使えるか確認。なければインストールを試みる"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def install_ffmpeg():
    """Streamlit Cloud (Debian/Ubuntu) に FFmpeg をインストール"""
    try:
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], capture_output=True)
        return check_ffmpeg()
    except Exception:
        return False


def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_audio(video_path, output_dir):
    audio_path = os.path.join(output_dir, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-y", audio_path],
        capture_output=True, check=True,
    )
    return audio_path


def extract_frames(video_path, output_dir, interval=10):
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    duration = get_video_duration(video_path)
    frames = []
    timestamps = list(range(0, int(duration), interval))
    if int(duration) not in timestamps:
        timestamps.append(int(duration) - 1)

    for i, ts in enumerate(timestamps):
        frame_filename = f"frame_{i:04d}_{ts}s.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ss", str(ts),
             "-frames:v", "1", "-q:v", "2", "-y", frame_path],
            capture_output=True, check=True,
        )
        frames.append({
            "index": i, "timestamp": ts,
            "filename": frame_filename, "path": frame_path,
        })
    return frames


def transcribe_audio_gemini(audio_path):
    """Gemini APIで音声を文字起こし（タイムスタンプ付き）"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # 音声ファイルを読み込み
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    prompt = """この音声ファイルを文字起こししてください。
以下のJSON形式で出力してください。コードブロックで囲わず、JSONのみ出力してください。

{
  "full_text": "全体のテキスト",
  "segments": [
    {"start": 0.0, "end": 5.0, "text": "セグメントのテキスト"},
    {"start": 5.0, "end": 10.0, "text": "次のセグメントのテキスト"}
  ]
}

注意:
- 日本語で文字起こししてください
- セグメントは内容のまとまりごとに区切ってください（10〜30秒程度）
- start/endは秒数です
- 必ず有効なJSONで出力してください"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=audio_data, mime_type="audio/wav"),
                    types.Part.from_text(text=prompt),
                ]
            )
        ],
    )

    # レスポンスからJSON部分を抽出
    response_text = response.text.strip()
    # コードブロックで囲まれている場合は除去
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # 最初と最後の ```行を除去
        lines = [l for l in lines if not l.strip().startswith("```")]
        response_text = "\n".join(lines)

    try:
        transcription = json.loads(response_text)
    except json.JSONDecodeError:
        # JSONパースに失敗した場合は、テキスト全体をフルテキストとして扱う
        transcription = {
            "full_text": response_text,
            "segments": [{"start": 0.0, "end": 0.0, "text": response_text}],
        }

    return transcription


def format_timestamp(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def generate_article(transcription, frames, video_filename, custom_prompt=""):
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    frames_info = "\n".join(
        f"  - {f['filename']} (タイムスタンプ: {format_timestamp(f['timestamp'])})"
        for f in frames
    )
    segments_text = "\n".join(
        f"[{format_timestamp(s['start'])} - {format_timestamp(s['end'])}] {s['text']}"
        for s in transcription["segments"]
    )

    # カスタムプロンプトがあれば追加要件として組み込む
    custom_section = ""
    if custom_prompt.strip():
        custom_section = f"""
## 追加の要件
{custom_prompt.strip()}
"""

    prompt = f"""以下は教材動画「{video_filename}」の文字起こしとキーフレーム画像の情報です。
これを元に、動画を見なくても手順や内容がわかる説明記事（Markdown形式）を作成してください。

## 要件
1. 記事のタイトルをつけてください
2. 冒頭に概要セクションを設けてください
3. 動画の流れに沿って、適切な見出し（##, ###）で章立てしてください
4. 各手順やポイントには、対応するスクリーンショットを挿入してください
   - 画像は `![説明](frames/ファイル名)` の形式で挿入
   - 全てのフレームを使う必要はありません。内容の変化がある重要な場面のフレームだけを厳選してください
   - 同じような画面・似たような内容のフレームは1枚だけ選び、重複して貼らないでください
   - 連続するフレーム（例: frame_0010 と frame_0011）は画面がほぼ同じなので、どちらか一方だけ使ってください
5. 手順がある場合は番号付きリストで記載してください
6. 補足情報やポイントは引用ブロック（>）やボールドで強調してください
7. 文字起こしの口語表現は、読みやすい文語表現に変換してください
{custom_section}
## 利用可能なフレーム画像
{frames_info}

## 文字起こし（タイムスタンプ付き）
{segments_text}

## 出力
Markdown形式の記事を出力してください。コードブロックで囲わず、そのままMarkdownを出力してください。"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def generate_preview_html(article_content, frames_dir):
    """HTMLプレビューを生成（画像をbase64で埋め込み、完全スタンドアロン）"""
    import markdown as md

    # 画像をbase64に変換してMarkdown内のパスを置換
    def replace_image_with_base64(match):
        alt = match.group(1)
        img_path = match.group(2)
        full_path = os.path.join(frames_dir, os.path.basename(img_path))
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f'![{alt}](data:image/jpeg;base64,{b64})'
        return match.group(0)

    # 画像パスをbase64に置換
    article_with_b64 = re.sub(
        r'!\[([^\]]*)\]\((frames/[^)]+)\)',
        replace_image_with_base64,
        article_content,
    )

    article_html = md.markdown(
        article_with_b64,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>記事プレビュー</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Hiragino Kaku Gothic ProN", sans-serif; line-height: 1.8; color: #333; background: #f5f5f5; }}
  .container {{ max-width: 800px; margin: 0 auto; padding: 40px 24px; background: #fff; min-height: 100vh; box-shadow: 0 0 20px rgba(0,0,0,0.05); }}
  h1 {{ font-size: 1.8em; color: #1a1a1a; border-bottom: 3px solid #2563eb; padding-bottom: 12px; margin-bottom: 24px; line-height: 1.4; }}
  h2 {{ font-size: 1.4em; color: #1a1a1a; margin-top: 48px; margin-bottom: 16px; padding-left: 12px; border-left: 4px solid #2563eb; }}
  h3 {{ font-size: 1.15em; color: #333; margin-top: 32px; margin-bottom: 12px; }}
  p {{ margin-bottom: 16px; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
  blockquote {{ background: #f0f7ff; border-left: 4px solid #2563eb; padding: 16px 20px; margin: 16px 0; border-radius: 0 8px 8px 0; }}
  blockquote p:last-child {{ margin-bottom: 0; }}
  ul, ol {{ margin: 12px 0; padding-left: 28px; }}
  li {{ margin-bottom: 8px; }}
  strong {{ color: #1a1a1a; }}
  hr {{ border: none; border-top: 1px solid #e5e5e5; margin: 40px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
  th {{ background: #f0f7ff; font-weight: bold; }}
  .header-bar {{ background: #2563eb; color: #fff; padding: 12px 24px; text-align: center; font-size: 0.85em; position: sticky; top: 0; z-index: 10; }}
</style>
</head>
<body>
<div class="header-bar">動画から自動生成された記事プレビュー</div>
<div class="container">
{article_html}
</div>
</body>
</html>'''
    return html


def display_article(article, frames_dir):
    """記事を画像参照で分割し、テキストはmarkdown、画像はst.imageで表示"""
    parts = re.split(r'(!\[[^\]]*\]\(frames/[^)]+\))', article)
    for part in parts:
        img_match = re.match(r'!\[([^\]]*)\]\(frames/([^)]+)\)', part)
        if img_match:
            alt = img_match.group(1)
            fname = img_match.group(2)
            frame_path = os.path.join(frames_dir, fname)
            if os.path.exists(frame_path):
                st.image(frame_path, caption=alt)
        elif part.strip():
            st.markdown(part, unsafe_allow_html=True)


# ========================================
# メイン画面
# ========================================

st.title("🎬 動画 → 説明記事 変換ツール")

# 事前チェック: FFmpeg
if not check_ffmpeg():
    with st.spinner("FFmpegをインストール中..."):
        if not install_ffmpeg():
            st.error("⚠️ FFmpegのインストールに失敗しました。")
            st.stop()

# 事前チェック: APIキー
gemini_key = os.environ.get("GEMINI_API_KEY", "")
if not gemini_key:
    st.warning("⚠️ Gemini APIキーが設定されていません。")
    st.markdown("管理者に連絡するか、[Google AI Studio](https://aistudio.google.com/apikey)でキーを取得してください。")
    st.stop()

# --- タブ ---
tab_convert, tab_guide = st.tabs(["📹 動画を変換", "📖 つかいかた"])

# ========================================
# タブ1: 動画を変換
# ========================================
with tab_convert:
    st.markdown("教材の動画ファイルをアップロードするだけで、**画像つきの説明記事**を自動生成します。")
    st.divider()

    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "動画ファイルをここにドラッグ＆ドロップ",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="対応形式: MP4, MOV, AVI, MKV, WebM",
    )

    # オプション
    with st.expander("⚙️ オプション設定", expanded=False):
        interval = st.slider(
            "スクリーンショットの間隔（秒）",
            min_value=3, max_value=30, value=10, step=1,
            help="小さい値にするほど画像が多くなります",
        )
        st.divider()
        st.markdown("**📝 追加プロンプト（AIへの追加指示）**")
        st.caption("記事の質を調整したいときに、自由に指示を追加できます")
        custom_prompt = st.text_area(
            "追加プロンプト",
            value="",
            height=100,
            placeholder="例：\n・箇条書きを多めにしてください\n・初心者向けにわかりやすく書いてください\n・各セクションの最後にポイントをまとめてください",
            label_visibility="collapsed",
        )

    # 変換開始
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / 1024 / 1024
        st.info(f"📁 **{uploaded_file.name}**（{file_size_mb:.1f}MB）")

        if st.button("🚀 記事に変換する", type="primary", use_container_width=True):

            # 一時ディレクトリに保存
            tmp_dir = tempfile.mkdtemp()
            tmp_video = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_video, "wb") as f:
                f.write(uploaded_file.getbuffer())

            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            video_name = Path(uploaded_file.name).stem
            progress = st.progress(0, text="準備中...")

            try:
                # Step 1: 音声抽出
                progress.progress(10, text="🔊 [1/4] 音声を抽出中...")
                audio_path = extract_audio(tmp_video, output_dir)

                # Step 2: フレーム抽出
                progress.progress(25, text="📸 [2/4] スクリーンショットを抽出中...")
                frames = extract_frames(tmp_video, output_dir, interval=interval)

                # Step 3: 文字起こし（Gemini）
                progress.progress(45, text="✍️ [3/4] 音声を文字起こし中（Gemini API）...")
                transcription = transcribe_audio_gemini(audio_path)

                # Step 4: 記事生成
                progress.progress(75, text="📝 [4/4] 記事を生成中...")
                article = generate_article(transcription, frames, video_name, custom_prompt)

                progress.progress(100, text="✅ 変換完了！")

                # --- 結果表示 ---
                st.balloons()
                st.success("🎉 記事が完成しました！")

                # HTMLダウンロードボタン
                frames_dir = os.path.join(output_dir, "frames")
                preview_html = generate_preview_html(article, frames_dir)

                st.download_button(
                    label="📥 記事をHTMLでダウンロード",
                    data=preview_html,
                    file_name=f"{video_name}_記事.html",
                    mime="text/html",
                    use_container_width=True,
                )

                st.caption("💡 ダウンロードしたHTMLファイルをダブルクリックで、画像付きの記事がブラウザで見れます")

                # Markdownダウンロード
                st.download_button(
                    label="📥 記事をMarkdownでダウンロード",
                    data=article,
                    file_name=f"{video_name}_記事.md",
                    mime="text/markdown",
                )

                # 記事プレビュー
                st.divider()
                st.subheader("📄 生成された記事")
                display_article(article, frames_dir)

                # 統計情報
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("セグメント数", f"{len(transcription['segments'])}")
                col2.metric("スクリーンショット", f"{len(frames)}枚")
                col3.metric("料金", "¥0")

            except Exception as e:
                progress.empty()
                st.error(f"エラーが発生しました: {str(e)}")
                st.exception(e)

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    else:
        st.markdown("""
        ### 使い方
        1. 上のエリアに **動画ファイルをドラッグ＆ドロップ**
        2. **「記事に変換する」** ボタンを押す
        3. 数分待つと記事が完成！
        4. **HTMLでダウンロード** → ダブルクリックで画像付き記事が見れる

        ### 完全無料で動作します
        - 🎤 文字起こし: Gemini API（無料枠）
        - 📝 記事生成: Gemini API（無料枠）
        """)

# ========================================
# タブ2: つかいかた
# ========================================
with tab_guide:
    if GUIDE_PATH.exists():
        guide_text = GUIDE_PATH.read_text(encoding="utf-8")
        st.markdown(guide_text)
    else:
        st.markdown("""
        ## 使い方

        1. **「動画を変換」タブ** を開く
        2. 動画ファイルをドラッグ＆ドロップ
        3. **「記事に変換する」** ボタンを押す
        4. 完成したら **HTMLでダウンロード**
        5. ダウンロードしたファイルをダブルクリックで記事が見れます

        ### 対応形式
        MP4, MOV, AVI, MKV, WebM

        ### 料金
        **完全無料** です（Gemini API無料枠を使用）
        """)

# フッター
st.divider()
st.caption("💡 文字起こし・記事生成: Gemini API（無料枠） / 料金: ¥0")
