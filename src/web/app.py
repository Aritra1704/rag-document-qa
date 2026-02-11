"""Streamlit UI for human-in-the-loop approval workflow."""

import requests
import streamlit as st

# Configuration
API_URL = "http://localhost:8000"


def init_session_state():
    """Initialize session state variables."""
    if "generated_posts" not in st.session_state:
        st.session_state.generated_posts = []
    if "last_action" not in st.session_state:
        st.session_state.last_action = None


def generate_post(topic: str, platform: str, tone: str, context: str):
    """Call the API to generate a new post."""
    try:
        response = requests.post(
            f"{API_URL}/posts/generate",
            json={
                "topic": topic,
                "platform": platform,
                "tone": tone,
                "additional_context": context,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to generate post: {e}")
        return None


def get_pending_posts():
    """Fetch all pending posts from the API."""
    try:
        response = requests.get(f"{API_URL}/posts/pending", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch pending posts: {e}")
        return []


def approve_post(thread_id: str, action: str, edited_text: str = None):
    """Send approval decision to the API."""
    try:
        payload = {"action": action}
        if edited_text:
            payload["edited_text"] = edited_text

        response = requests.post(
            f"{API_URL}/posts/{thread_id}/approve",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to process approval: {e}")
        return None


def render_post_card(post: dict, index: int):
    """Render a single post card with approval actions."""
    with st.container():
        st.markdown(f"### 📝 Post #{index + 1}")

        # Platform badge
        platform = post.get("platform", "twitter").upper()
        platform_color = "🐦" if platform == "TWITTER" else "💼"
        st.markdown(f"{platform_color} **{platform}** | Topic: *{post.get('topic', 'N/A')}*")

        # Character count indicator
        char_count = post.get("char_count", 0)
        max_chars = 280 if platform == "TWITTER" else 3000
        char_pct = (char_count / max_chars) * 100

        if char_pct > 90:
            st.progress(char_pct / 100, text=f"⚠️ {char_count}/{max_chars} characters")
        else:
            st.progress(char_pct / 100, text=f"✅ {char_count}/{max_chars} characters")

        # Post content
        st.markdown("---")
        post_text = post.get("post_text", "")
        st.markdown(f"> {post_text}")
        st.markdown("---")

        # Action buttons
        col1, col2, col3 = st.columns(3)

        thread_id = post.get("thread_id", "")

        with col1:
            if st.button("✅ Approve", key=f"approve_{thread_id}", type="primary"):
                result = approve_post(thread_id, "approve")
                if result:
                    st.success(f"Post approved! Status: {result.get('status')}")
                    if result.get("published_url"):
                        st.markdown(f"🔗 [View Post]({result.get('published_url')})")
                    st.session_state.last_action = "approved"
                    st.rerun()

        with col2:
            if st.button("❌ Reject", key=f"reject_{thread_id}"):
                result = approve_post(thread_id, "reject")
                if result:
                    st.warning("Post rejected. Regenerating...")
                    st.session_state.last_action = "rejected"
                    st.rerun()

        with col3:
            if st.button("✏️ Edit", key=f"edit_{thread_id}"):
                st.session_state[f"editing_{thread_id}"] = True

        # Edit mode
        if st.session_state.get(f"editing_{thread_id}", False):
            edited_text = st.text_area(
                "Edit your post:",
                value=post_text,
                key=f"edit_text_{thread_id}",
                max_chars=max_chars,
            )

            if st.button("💾 Save & Approve", key=f"save_{thread_id}"):
                result = approve_post(thread_id, "edit", edited_text)
                if result:
                    st.success("Post edited and approved!")
                    if result.get("published_url"):
                        st.markdown(f"🔗 [View Post]({result.get('published_url')})")
                    st.session_state[f"editing_{thread_id}"] = False
                    st.session_state.last_action = "edited"
                    st.rerun()

        st.markdown("---")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Social Media AI Agent",
        page_icon="🤖",
        layout="wide",
    )

    init_session_state()

    # Header
    st.title("🤖 Social Media AI Agent")
    st.markdown("Generate AI-powered social media posts with human approval workflow")

    # Sidebar - Generate new post
    with st.sidebar:
        st.header("📝 Generate New Post")

        topic = st.text_input("Topic", placeholder="e.g., AI in healthcare")

        platform = st.selectbox(
            "Platform",
            options=["twitter"],
            format_func=lambda x: "🐦 Twitter/X" if x == "twitter" else x,
        )

        tone = st.selectbox(
            "Tone",
            options=["professional", "casual", "humorous", "inspirational", "educational"],
        )

        context = st.text_area(
            "Additional Context",
            placeholder="Any specific requirements or hashtags to include...",
        )

        if st.button("🚀 Generate Post", type="primary", use_container_width=True):
            if not topic:
                st.error("Please enter a topic")
            else:
                with st.spinner("Generating post..."):
                    result = generate_post(topic, platform, tone, context)
                    if result:
                        st.success("Post generated! Check the main panel.")
                        st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.text_input("API URL", value=API_URL, disabled=True)

    # Main content - Pending posts
    st.header("📋 Pending Approvals")

    # Refresh button
    if st.button("🔄 Refresh"):
        st.rerun()

    # Fetch and display pending posts
    pending_posts = get_pending_posts()

    if not pending_posts:
        st.info(
            "No posts pending approval. Use the sidebar to generate a new post!"
        )
    else:
        st.markdown(f"**{len(pending_posts)} post(s) awaiting your approval**")

        for i, post in enumerate(pending_posts):
            render_post_card(post, i)

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using LangGraph, FastAPI, and Streamlit | "
        "[GitHub](https://github.com/yourusername/social-media-ai-agent)"
    )


if __name__ == "__main__":
    main()
