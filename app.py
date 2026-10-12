"""
Streamlit Web Application for Wordle AI Project.
Provides interactive visual gameplay, live candidate reduction tracking,
neuro-symbolic score breakdowns, real-game human assistant mode, and research benchmark figures.
"""

import os
import random
import streamlit as st
import pandas as pd
from PIL import Image

from src.wordle.dictionary import WordleDictionary
from src.wordle.game import WordleGame, GameStatus
from src.wordle.constraints import SymbolicConstraints
from src.wordle.feedback import feedback_to_emoji, FeedbackTile, get_feedback
from src.solvers.random_solver import RandomSolver
from src.solvers.frequency_solver import FrequencySolver
from src.solvers.entropy_solver import EntropySolver
from src.solvers.character_transformer_solver import CharacterTransformerSolver
from src.solvers.hybrid_solver import HybridSolver

st.set_page_config(
    page_title="Wordle AI Research Lab",
    page_icon="🟩",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_dictionary():
    return WordleDictionary(seed=42)

dict_mgr = load_dictionary()

@st.cache_resource
def get_solver(name: str):
    if name == "Entropy Solver":
        return EntropySolver(dict_mgr)
    elif name == "Character Transformer":
        return CharacterTransformerSolver(dict_mgr)
    elif name == "Letter Frequency":
        return FrequencySolver(dict_mgr)
    elif name == "Random":
        return RandomSolver(dict_mgr)
    else:
        return HybridSolver(dict_mgr)

st.title("Wordle AI — Neuro-Symbolic & Transformer Solver Lab")
st.markdown(
    "Explore how **Classical Information Theory**, **Character Transformers**, "
    "and **Neuro-Symbolic Constraint Solvers** master Wordle."
)

tab1, tab2, tab3 = st.tabs([
    "Interactive AI Game",
    "Live Wordle Assistant",
    "Research Benchmark Gallery"
])

# ==============================================================================
# TAB 1: INTERACTIVE AI GAME
# ==============================================================================
with tab1:
    st.sidebar.header("Game Controls")
    model_choice = st.sidebar.selectbox(
        "Select AI Solver Model",
        ["Neuro-Symbolic Hybrid", "Entropy Solver", "Character Transformer", "Letter Frequency", "Random"],
        key="model_select"
    )

    solver = get_solver(model_choice)

    # Initialize secret word in session state
    if "target_secret" not in st.session_state:
        st.session_state.target_secret = "apple"

    # Random secret button
    if st.sidebar.button("🎲 Random Secret Word"):
        st.session_state.target_secret = random.choice(dict_mgr.test_secrets[:100])
        st.session_state.reset_requested = True

    secret_input = st.sidebar.text_input(
        "Secret Target Word (5 letters)",
        value=st.session_state.target_secret,
        key="secret_input_box"
    ).lower().strip()

    if len(secret_input) == 5 and secret_input.isalpha():
        if secret_input != st.session_state.target_secret:
            st.session_state.target_secret = secret_input
            st.session_state.reset_requested = True
    else:
        st.sidebar.warning("Secret word must be exactly 5 alphabetic letters.")
        secret_input = st.session_state.target_secret

    # Check if a game reset is needed
    if (
        "game" not in st.session_state
        or st.session_state.get("reset_requested", False)
        or st.sidebar.button("🎮 Start New Game", key="btn_start_new")
    ):
        st.session_state.game = WordleGame(secret_word=st.session_state.target_secret, dictionary=dict_mgr)
        st.session_state.constraints = SymbolicConstraints()
        st.session_state.history_cands = [len(dict_mgr.secret_words)]
        st.session_state.reset_requested = False

    game = st.session_state.game
    constraints = st.session_state.constraints

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Wordle Game Board")

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        
        if game.status == GameStatus.IN_PROGRESS:
            with btn_col1:
                if st.button("🤖 Play Next Turn", type="primary", use_container_width=True):
                    with st.spinner("Calculating optimal guess..."):
                        turn = game.turns_used + 1
                        guess = solver.choose_guess(constraints, turn=turn)
                        fb = game.make_guess(guess)
                        constraints.add_feedback(guess, fb)
                        rem = len(constraints.filter_candidates(dict_mgr.secret_words))
                        st.session_state.history_cands.append(rem)
                    st.rerun()

            with btn_col2:
                if st.button("⚡ Auto-Solve Game", use_container_width=True):
                    with st.spinner("AI is solving the game..."):
                        while game.status == GameStatus.IN_PROGRESS and game.turns_used < 6:
                            turn = game.turns_used + 1
                            guess = solver.choose_guess(constraints, turn=turn)
                            fb = game.make_guess(guess)
                            constraints.add_feedback(guess, fb)
                            rem = len(constraints.filter_candidates(dict_mgr.secret_words))
                            st.session_state.history_cands.append(rem)
                    st.rerun()
        else:
            with btn_col1:
                if st.button("🔄 Play Again", type="primary", use_container_width=True):
                    st.session_state.target_secret = random.choice(dict_mgr.test_secrets[:100])
                    st.session_state.reset_requested = True
                    st.rerun()

        # Display Board Rows
        for turn_idx in range(6):
            cols = st.columns(5)
            if turn_idx < len(game.guesses):
                guess_word = game.guesses[turn_idx].upper()
                fb = game.feedbacks[turn_idx]
                for letter_idx in range(5):
                    char = guess_word[letter_idx]
                    tile_type = fb[letter_idx]
                    bg_color = "#6aaa64" if tile_type == 2 else ("#c9b458" if tile_type == 1 else "#787c7e")
                    cols[letter_idx].markdown(
                        f"<div style='text-align:center; background-color:{bg_color}; color:white; "
                        f"font-size:26px; font-weight:bold; padding:12px 0; border-radius:6px; margin:2px;'>{char}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                for letter_idx in range(5):
                    cols[letter_idx].markdown(
                        "<div style='text-align:center; background-color:#3a3a3c; color:white; "
                        "font-size:26px; font-weight:bold; padding:12px 0; border-radius:6px; margin:2px;'> </div>",
                        unsafe_allow_html=True,
                    )

        if game.status == GameStatus.WON:
            st.balloons()
            st.success(f"🎉 SOLVED! {model_choice} identified '{st.session_state.target_secret.upper()}' in {game.turns_used} turns!")
        elif game.status == GameStatus.LOST:
            st.error(f"❌ Game Over! The secret word was: {st.session_state.target_secret.upper()}")

    with col2:
        st.subheader("Model Explainability & Candidate Space")

        rem_candidates = constraints.filter_candidates(dict_mgr.secret_words)
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Candidates Remaining", len(rem_candidates))
        m_col2.metric("Turns Used", f"{game.turns_used} / 6")

        if isinstance(solver, HybridSolver) and rem_candidates and len(rem_candidates) > 1:
            st.markdown("##### Top Candidate Word Score Breakdown")
            rankings = solver.get_candidate_rankings(constraints, top_k=8)
            if rankings:
                df = pd.DataFrame(rankings)
                st.dataframe(df, use_container_width=True)

        # Candidate reduction chart
        st.markdown("##### Candidate Space Reduction Curve")
        chart_data = pd.DataFrame({
            "Turn": list(range(len(st.session_state.history_cands))),
            "Candidates": st.session_state.history_cands
        })
        st.line_chart(chart_data.set_index("Turn"))


# ==============================================================================
# TAB 2: LIVE WORDLE ASSISTANT (FOR HUMAN PLAYERS)
# ==============================================================================
with tab2:
    st.subheader("Live Wordle Assistant for Real Games")
    st.markdown(
        "Playing Wordle on NYT or an external app? Enter the words you played and their tile feedback colors below, "
        "and our AI will calculate the optimal next guesses!"
    )

    if "assistant_constraints" not in st.session_state:
        st.session_state.assistant_constraints = SymbolicConstraints()
        st.session_state.assistant_history = []

    asst_constraints = st.session_state.assistant_constraints

    with st.expander("How Feedback Input Works", expanded=False):
        st.markdown("""
        Enter feedback as 5 digits:
        - `2` = Green (Correct letter & spot)
        - `1` = Yellow (Correct letter, wrong spot)
        - `0` = Gray (Letter not in word)
        
        *Example:* If you guessed `SLATE` and got Gray, Yellow, Gray, Green, Gray, enter `01020`.
        """)

    with st.form("assistant_form"):
        f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1])
        with f_col1:
            input_guess = st.text_input("Your Guess Word", max_chars=5, placeholder="e.g. slate").lower().strip()
        with f_col2:
            input_fb = st.text_input("Feedback Pattern (5 digits)", max_chars=5, placeholder="e.g. 01020").strip()
        with f_col3:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Add Clue", type="primary", use_container_width=True)

    if submitted:
        if len(input_guess) == 5 and input_guess.isalpha() and len(input_fb) == 5 and all(c in "012" for c in input_fb):
            fb_tuple = tuple(int(c) for c in input_fb)
            asst_constraints.add_feedback(input_guess, fb_tuple)
            st.session_state.assistant_history.append((input_guess, fb_tuple))
            st.success(f"Added clue: {input_guess.upper()} -> {feedback_to_emoji(fb_tuple)}")
        else:
            st.error("Please enter a valid 5-letter word and 5 digits containing only 0, 1, or 2.")

    if st.button("Reset Assistant"):
        st.session_state.assistant_constraints = SymbolicConstraints()
        st.session_state.assistant_history = []
        st.rerun()

    # Display clues entered so far
    if st.session_state.assistant_history:
        st.markdown("##### Clues Recorded:")
        for w, f in st.session_state.assistant_history:
            st.markdown(f"- **{w.upper()}**: {feedback_to_emoji(f)}")

    # Remaining candidates and AI recommendation
    filtered_words = asst_constraints.filter_candidates(dict_mgr.secret_words)
    st.metric("Compatible Solutions Left", len(filtered_words))

    if filtered_words:
        st.markdown("##### Top AI Recommendations for Your Next Guess:")
        asst_hybrid = get_solver("Neuro-Symbolic Hybrid")
        recs = asst_hybrid.get_candidate_rankings(asst_constraints, top_k=10)
        if recs:
            df_recs = pd.DataFrame(recs)
            st.dataframe(df_recs, use_container_width=True)
        else:
            st.write(", ".join([w.upper() for w in filtered_words[:20]]))


# ==============================================================================
# TAB 3: RESEARCH BENCHMARK GALLERY
# ==============================================================================
with tab3:
    st.subheader("Research Benchmark Gallery")
    st.markdown(
        "Publication-quality evaluation plots comparing **Classical Information Theory**, "
        "**Character-Level Transformer**, **Tiny Pretrained LM**, and **Neuro-Symbolic Hybrid** on 265 held-out test secret words."
    )

    fig_dir = "reports/figures"
    fig_files = [
        ("win_rate_comparison.png", "Win Rate Comparison across All Solvers", "Comparison of percentage of games won within 6 turns."),
        ("avg_guesses_comparison.png", "Average Guesses to Win (Lower is Better)", "Mean number of guesses required to identify the secret word."),
        ("guess_distribution.png", "Guess Distribution Histogram", "Distribution of game resolutions across turns 1 through 6, and losses."),
        ("candidate_reduction_curve.png", "Candidate Space Reduction by Turn", "Log-scale reduction of remaining valid candidate words per turn."),
        ("latency_vs_accuracy.png", "Inference Latency vs. Accuracy Tradeoff", "Scatter comparison of turn latency (ms) vs. win rate (%).")
    ]

    for fname, title, desc in fig_files:
        fpath = os.path.join(fig_dir, fname)
        if os.path.exists(fpath):
            st.markdown(f"#### {title}")
            st.caption(desc)
            image = Image.open(fpath)
            st.image(image, use_container_width=True)
            st.divider()
        else:
            st.info(f"Figure '{fname}' will appear here once generated via `python scripts/benchmark_all.py`.")
