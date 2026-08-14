"""
Streamlit Web Application for Wordle AI Project.
Provides interactive visual gameplay, live candidate reduction tracking, and neuro-symbolic score breakdowns.
"""

import streamlit as st
import pandas as pd
from src.wordle.dictionary import WordleDictionary
from src.wordle.game import WordleGame, GameStatus
from src.wordle.constraints import SymbolicConstraints
from src.wordle.feedback import feedback_to_emoji, FeedbackTile
from src.solvers.random_solver import RandomSolver
from src.solvers.frequency_solver import FrequencySolver
from src.solvers.entropy_solver import EntropySolver
from src.solvers.character_transformer_solver import CharacterTransformerSolver
from src.solvers.hybrid_solver import HybridSolver

st.set_page_config(page_title="Wordle AI Research Lab", page_icon="🟩", layout="wide")

@st.cache_resource
def load_dictionary():
    return WordleDictionary(seed=42)

dict_mgr = load_dictionary()

st.title("🧩 Wordle AI — Neuro-Symbolic & Transformer Solver Lab")
st.markdown(
    "Investigating **Classical Information Theory**, **Character Transformers**, "
    "**Tiny Pretrained LMs**, and **Neuro-Symbolic Constraint Solvers** for Wordle."
)

st.sidebar.header("🕹️ Control Panel")
model_choice = st.sidebar.selectbox(
    "Select AI Solver Model",
    ["Neuro-Symbolic Hybrid", "Entropy Solver", "Character Transformer", "Letter Frequency", "Random"]
)

secret_input = st.sidebar.text_input("Secret Target Word (5 letters)", value="crane").lower()

if len(secret_input) != 5 or not secret_input.isalpha():
    st.sidebar.warning("Secret word must be exactly 5 alphabetic letters.")
    secret_input = "crane"

# Instantiate Solver
if model_choice == "Entropy Solver":
    solver = EntropySolver(dict_mgr)
elif model_choice == "Character Transformer":
    solver = CharacterTransformerSolver(dict_mgr)
elif model_choice == "Letter Frequency":
    solver = FrequencySolver(dict_mgr)
elif model_choice == "Random":
    solver = RandomSolver(dict_mgr)
else:
    solver = HybridSolver(dict_mgr)

# Initialize Session State
if "game" not in st.session_state or st.sidebar.button("🎮 Start New Game"):
    st.session_state.game = WordleGame(secret_word=secret_input, dictionary=dict_mgr)
    st.session_state.constraints = SymbolicConstraints()
    st.session_state.history_cands = [len(dict_mgr.secret_words)]

game = st.session_state.game
constraints = st.session_state.constraints

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🎯 Wordle Game Board")
    
    # Action button
    if game.status == GameStatus.IN_PROGRESS:
        if st.button("🤖 Let AI Play Next Turn"):
            turn = game.turns_used + 1
            guess = solver.choose_guess(constraints, turn=turn)
            fb = game.make_guess(guess)
            constraints.add_feedback(guess, fb)
            rem = len(constraints.filter_candidates(dict_mgr.secret_words))
            st.session_state.history_cands.append(rem)
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
                    f"<div style='text-align:center; background-color:{bg_color}; color:white; font-size:24px; font-weight:bold; padding:10px; border-radius:5px;'>{char}</div>",
                    unsafe_allow_html=True,
                )
        else:
            for letter_idx in range(5):
                cols[letter_idx].markdown(
                    f"<div style='text-align:center; background-color:#3a3a3c; color:white; font-size:24px; font-weight:bold; padding:10px; border-radius:5px;'> </div>",
                    unsafe_allow_html=True,
                )

    if game.status == GameStatus.WON:
        st.balloons()
        st.success(f"🎉 SOLVED! The AI won in {game.turns_used} turns!")
    elif game.status == GameStatus.LOST:
        st.error(f"❌ Game Over! Secret word was: {secret_input.upper()}")

with col2:
    st.subheader("🔍 Neuro-Symbolic Explainability & Ranking")
    
    rem_candidates = constraints.filter_candidates(dict_mgr.secret_words)
    st.metric("Candidates Remaining", len(rem_candidates))

    if isinstance(solver, HybridSolver) and rem_candidates and len(rem_candidates) > 1:
        st.markdown("##### Top Candidate Word Score Breakdown")
        rankings = solver.get_candidate_rankings(constraints, top_k=8)
        if rankings:
            df = pd.DataFrame(rankings)
            st.dataframe(df, use_container_width=True)

    # Candidate reduction chart
    st.markdown("##### Candidate Space Reduction Curve")
    chart_data = pd.DataFrame({"Turn": list(range(len(st.session_state.history_cands))), "Candidates": st.session_state.history_cands})
    st.line_chart(chart_data.set_index("Turn"))
