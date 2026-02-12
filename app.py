import streamlit as st
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import DOWNLOAD_DIR, OUTPUT_DIR, ENABLE_CACHE, OLLAMA_MODEL
from src.data_loader import load_data
from src.downloader import download_file, batch_download
from src.extractor import process_file, detect_file_type
from src.evaluator import evaluate_submission, batch_evaluate, evaluate_architecture
from src.visual_scorer import VisualScorer
from src.multi_agent_evaluator import MultiAgentEvaluator

# Page Config
st.set_page_config(
    page_title="Hackathon Ranker AI", 
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Gradient Title */
        .title-text {
            background: linear-gradient(45deg, #FF4B2B, #FF416C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
        }
        
        /* Cards */
        .stMetric {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Tables */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #111;
            border-right: 1px solid #222;
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
            color: white;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
            border-radius: 30px;
            transition: transform 0.2s;
        }
        .stButton button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# Main Title with Gradient
st.markdown('<h1 class="title-text">🏆 Hackathon Ranker AI</h1>', unsafe_allow_html=True)
st.markdown("Automated judge using **local LLMs (Qwen2.5) + OCR**. ⚡ Optimized for privacy & speed.")

# 1. Settings Sidebar
st.sidebar.header("⚙️ Configuration")
try:
    model_name_display = OLLAMA_MODEL.split(':')[-1]  # Get model version
except:
    model_name_display = OLLAMA_MODEL
st.sidebar.success(f"🤖 Model: **{model_name_display}**")
st.sidebar.info("Lightweight model. Runs smoothly on 8GB RAM.")

# Performance Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Performance")
enable_cache = st.sidebar.checkbox("Enable Caching", value=ENABLE_CACHE, help="Skip re-evaluating processed teams")
parallel_extract = st.sidebar.checkbox("Parallel Extraction", value=True, help="Speed up file processing")

# 2. Main Input Section
st.markdown("### 1. 📂 Upload Team Data")

col1, col2 = st.columns([1, 2])
with col1:
    input_option = st.radio("Select Input Source:", ["Upload CSV/Excel", "Use Demo Data"], label_visibility="collapsed")

df = None
if input_option == "Upload CSV/Excel":
    uploaded_file = st.file_uploader("Drop your spreadsheet here", type=["csv", "xlsx"])
    if uploaded_file:
        if not os.path.exists("data"):
            os.makedirs("data")
        temp_path = os.path.join("data", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df, err = load_data(temp_path)
        if err:
            st.error(err)
else:
    # Detailed demo data
    df = pd.DataFrame([
        {
            "Team Name": "Team Alpha",
            "Team Leader": "Alice",
            "PPT Link": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", 
            "Problem Statement": "Sustainable Energy Monitor"
        },
        {
            "Team Name": "Team Beta",
            "Team Leader": "Bob",
            "PPT Link": "https://file-examples.com/storage/fe555ad819634d5464a9193/2017/10/file-sample_150kB.pdf",
            "Problem Statement": "AI Healthcare Assistant"
        },
        {
            "Team Name": "Team Gamma",
            "Team Leader": "Charlie",
            "PPT Link": "https://www.africau.edu/images/default/sample.pdf",
            "Problem Statement": "FinTech Fraud Detection"
        }
    ])
    from src.data_loader import clean_drive_link
    df['Download Link'] = df['PPT Link'].apply(clean_drive_link)
    st.info("✅ Loaded demo data with 3 sample teams.")

if df is not None:
    with st.expander("👀 View Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Execution
    if st.button("🚀 Start AI Evaluation", type="primary"):
        start_time = time.time()
        
        # Layout for live stats
        c1, c2, c3 = st.columns(3)
        stat_teams = c1.metric("Teams", len(df))
        stat_status = c2.empty()
        stat_time = c3.empty()
        
        progress_bar = st.progress(0)
        
        # ============= PHASE 1: DOWNLOAD =============
        stat_status.info("📥 Downloading presentations...")
        team_data = df.to_dict('records')
        download_map = batch_download(team_data)
        progress_bar.progress(0.2)
        
        total_teams = len(team_data)
        download_time = time.time() - start_time
        
        # ============= PHASE 2: PARALLEL EXTRACTION =============
        stat_status.info(f"📄 Extracting content from {total_teams} presentations...")
        extract_start = time.time()
        
        v_scorer = VisualScorer()
        submissions = []
        extraction_data = {}
        
        def extract_single(row):
            """Extract content and visual data for a single team."""
            team_name = row['Team Name']
            problem = row.get('Problem Statement', "General Project")
            local_path = download_map.get(team_name)
            
            result = {
                "team_name": team_name,
                "problem": problem,
                "local_path": local_path,
                "content": None,
                "visual_data": None,
                "visual_context": ""
            }
            
            if local_path and os.path.exists(local_path):
                content = process_file(local_path)
                ftype = detect_file_type(local_path)
                visual_data = v_scorer.evaluate(local_path, ftype)
                
                result["content"] = content
                result["visual_data"] = visual_data
                result["visual_context"] = f"Visual:{visual_data.get('visual_score',0)}/15 {visual_data.get('feedback','')[:100]}"
            
            return result
        
        if parallel_extract:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(extract_single, row): row['Team Name'] for row in team_data}
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    progress_bar.progress(0.2 + 0.3 * (completed / total_teams))
                    try:
                        result = future.result()
                        extraction_data[result["team_name"]] = result
                        if result["content"]:
                            submissions.append({
                                "team_name": result["team_name"],
                                "ppt_text": result["content"],
                                "visual_context": result["visual_context"]
                            })
                    except Exception as e:
                        st.warning(f"Error extracting {futures[future]}: {e}")
        else:
            for i, row in enumerate(team_data):
                result = extract_single(row)
                extraction_data[result["team_name"]] = result
                if result["content"]:
                    submissions.append({
                        "team_name": result["team_name"],
                        "ppt_text": result["content"],
                        "visual_context": result["visual_context"]
                    })
                progress_bar.progress(0.2 + 0.3 * ((i + 1) / total_teams))
        
        extract_time = time.time() - extract_start
        stat_status.info(f"🤖 AI Evaluation: Judging {len(submissions)} teams...")
        progress_bar.progress(0.5)
        
        # ============= PHASE 3: MULTI-AGENT AI EVALUATION =============
        eval_start = time.time()
        
        all_eval_results = {}
        
        if len(submissions) > 0:
            # Initialize multi-agent evaluator
            evaluator = MultiAgentEvaluator(model_name=OLLAMA_MODEL)
            
            # Evaluate each submission
            for i, sub in enumerate(submissions):
                team_name = sub['team_name']
                row = next((r for r in team_data if r['Team Name'] == team_name), {})
                problem_statement = row.get('Problem Statement', 'General Project')
                
                print(f"  Evaluating {team_name} ({i+1}/{len(submissions)})...")
                
                # Call multi-agent evaluator
                result = evaluator.evaluate(
                    problem_statement=problem_statement,
                    ppt_content=sub['ppt_text'],
                    visual_analysis=sub.get('visual_context', '')
                )
                
                all_eval_results[team_name] = result
                progress_bar.progress(0.5 + 0.4 * ((i + 1) / len(submissions)))
        
        eval_time = time.time() - eval_start
        
        # ============= PHASE 4: COMPILE RESULTS =============
        stat_status.info("📊 Compiling final leaderboard...")
        results = []
        
        for row in team_data:
            team_name = row['Team Name']
            problem = row.get('Problem Statement', "General Project")
            
            score_data = {
                "Team Name": team_name,
                "Problem Statement": problem,
                "Total Score": 0,
                "Rank Reason": "Not processed"
            }
            
            ext = extraction_data.get(team_name, {})
            eval_result = all_eval_results.get(team_name, {})
            visual_data = ext.get("visual_data", {})
            
            if eval_result:
                # Normalize from 140-point to 100-point scale
                final_score = eval_result.get("final_score", 0)
                normalized_score = round((final_score / 140) * 100, 1)
                
                score_data["Total Score"] = normalized_score
                score_data["Visual Score"] = visual_data.get("visual_score", 0) if visual_data else 0
                score_data["Score Category"] = eval_result.get("score_category", "N/A")
                score_data["Confidence"] = eval_result.get("confidence", {}).get("level", "N/A")
                
                # Judge breakdown (normalized to /100)
                judge_scores = eval_result.get("judge_scores", {})
                score_data["Technical Score"] = round((judge_scores.get("technical", 0) / 84) * 60, 1)
                score_data["Product Score"] = round((judge_scores.get("product", 0) / 35) * 25, 1)
                score_data["Execution Score"] = round((judge_scores.get("execution", 0) / 21) * 15, 1)
                
                # Summary for rank reason
                insights = eval_result.get("aggregated_insights", {})
                strengths = insights.get("top_strengths", [])
                if strengths:
                    score_data["Rank Reason"] = strengths[0][:100]
                else:
                    score_data["Rank Reason"] = eval_result.get("score_category", "Evaluated")
                
                # Store full eval result for detailed view
                score_data["_eval_details"] = eval_result
                    
            elif ext.get("content"):
                score_data["Rank Reason"] = "Evaluation failed"
            else:
                score_data["Rank Reason"] = "Download/Extraction failed"
            
            results.append(score_data)
        
        # Complete
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        stat_status.success("Done!")
        stat_time.text(f"⏱️ {total_time:.1f}s")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="Total Score", ascending=False)
        results_df.insert(0, 'Rank', range(1, 1 + len(results_df)))
        
        # ============= SHOW RESULTS =============
        st.markdown("### 🏆 Final Leaderboard")
        
        # Modern Dataframe Configuration
        column_config = {
            "Rank": st.column_config.NumberColumn("Rank", format="No. %d", width="small"),
            "Team Name": st.column_config.TextColumn("Team", width="medium"),
            "Total Score": st.column_config.ProgressColumn(
                "Total Score", 
                format="%d", 
                min_value=0, 
                max_value=100,
                width="medium"
            ),
            "Visual Score": st.column_config.ProgressColumn(
                "Visual", 
                format="%.1f", 
                min_value=0, 
                max_value=15,
                width="small"
            ),
            "Problem Statement": st.column_config.TextColumn("Problem", width="large"),
        }
        
        # Hide internal detail columns from main view
        display_columns = ['Rank', 'Team Name', 'Total Score', 'Technical Score', 'Product Score', 'Execution Score', 'Confidence', 'Score Category']
        
        # Add columns to config
        column_config["Technical Score"] = st.column_config.NumberColumn("Tech", format="%.1f/60", width="small")
        column_config["Product Score"] = st.column_config.NumberColumn("Product", format="%.1f/25", width="small")
        column_config["Execution Score"] = st.column_config.NumberColumn("Exec", format="%.1f/15", width="small")
        column_config["Confidence"] = st.column_config.TextColumn("Confidence", width="small")
        column_config["Score Category"] = st.column_config.TextColumn("Category", width="medium")
        
        display_df = results_df[display_columns]
        
        st.dataframe(
            display_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        
        # Podium
        st.markdown("---")
        st.subheader("🥇 Podium Finishers")
        cols = st.columns(min(3, len(results_df)))
        medals = ["🥇 Gold", "🥈 Silver", "🥉 Bronze"]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
        
        for idx, (_, row) in enumerate(results_df.head(3).iterrows()):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background-color: {colors[idx]}22; padding: 20px; border-radius: 10px; border: 1px solid {colors[idx]}; text-align: center;">
                        <h3 style="color: {colors[idx]}; margin:0;">{medals[idx]}</h3>
                        <h4 style="margin: 10px 0;">{row['Team Name']}</h4>
                        <h1 style="margin: 0; font-size: 3rem;">{row['Total Score']}</h1>
                        <p style="opacity: 0.8; font-size: 0.9rem;">{row.get('Rank Reason', '')[:80]}...</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # ============= DETAILED FEEDBACK SECTIONS =============
        st.markdown("---")
        st.markdown("### 📋 Detailed Evaluation Reports")
        st.info("Expand each team to see criteria-level scores, strengths, weaknesses, and improvement suggestions from all judges.")
        
        for _, row in results_df.iterrows():
            team_name = row['Team Name']
            eval_details = row.get('_eval_details', {})
            
            if not eval_details:
                continue
                
            # Color code based on score
            score = row['Total Score']
            if score >= 70:
                emoji = "🟢"
            elif score >= 40:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            with st.expander(f"{emoji} **{team_name}** - {score}/100 ({row.get('Score Category', 'N/A')})"):
                # Overall Summary
                st.markdown(f"**Confidence:** {row.get('Confidence', 'N/A')} | **Category:** {row.get('Score Category', 'N/A')}")
                
                # Judge Details
                judge_details = eval_details.get('judge_details', {})
                variance = eval_details.get('variance_analysis', {})
                insights = eval_details.get('aggregated_insights', {})
                
                # Show variance analysis if high disagreement
                if variance.get('high_disagreement'):
                    st.warning(f"⚠️ **Judge Disagreement Detected**: {variance.get('disagreement_interpretation', 'N/A')}")
                
                # Top Strengths
                st.markdown("#### 💪 Key Strengths")
                strengths = insights.get('top_strengths', [])
                if strengths:
                    for strength in strengths[:5]:
                        st.markdown(f"- {strength}")
                else:
                    st.markdown("_No strengths identified_")
                
                # Critical Weaknesses
                st.markdown("#### ⚠️ Critical Weaknesses")
                weaknesses = insights.get('critical_weaknesses', [])
                if weaknesses:
                    for weakness in weaknesses[:5]:
                        st.markdown(f"- {weakness}")
                else:
                    st.markdown("_No major weaknesses identified_")
                
                # Priority Improvements
                st.markdown("#### 🔧 Priority Improvements")
                improvements = insights.get('priority_improvements', [])
                if improvements:
                    for i, improvement in enumerate(improvements, 1):
                        st.markdown(f"{i}. {improvement}")
                else:
                    st.markdown("_No specific improvements suggested_")
                
                st.markdown("---")
                
                # Technical Judge Details
                if 'technical' in judge_details:
                    tech = judge_details['technical']
                    st.markdown("### 🔧 Technical Judge")
                    
                    tech_scores = tech.get('technical_scores', {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Architecture", f"{tech_scores.get('architecture_quality', 0)}/30")
                    with col2:
                        st.metric("Algorithm", f"{tech_scores.get('algorithm_justification', 0)}/15")
                    with col3:
                        st.metric("Trade-offs", f"{tech_scores.get('trade_offs', 0)}/10")
                    with col4:
                        st.metric("Realism", f"{tech_scores.get('engineering_realism', 0)}/5")
                    
                    tech_reasoning = tech.get('technical_reasoning', {})
                    
                    # Defensive: ensure reasoning is a dict
                    if not isinstance(tech_reasoning, dict):
                        tech_reasoning = {}
                    
                    if tech_reasoning.get('technical_strengths'):
                        st.markdown("**✅ Technical Strengths:**")
                        for s in tech_reasoning['technical_strengths']:
                            st.markdown(f"- {s}")
                    
                    if tech_reasoning.get('critical_gaps'):
                        st.markdown("**❌ Technical Gaps:**")
                        for g in tech_reasoning['critical_gaps']:
                            st.markdown(f"- {g}")
                    
                    if tech_reasoning.get('improvements'):
                        st.markdown("**🔨 Technical Improvements:**")
                        for imp in tech_reasoning['improvements']:
                            st.markdown(f"- {imp}")
                
                # Product Judge Details
                if 'product' in judge_details:
                    prod = judge_details['product']
                    st.markdown("### 💡 Product & Innovation Judge")
                    
                    prod_scores = prod.get('product_scores', {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Innovation", f"{prod_scores.get('innovation_score', 0)}/20")
                    with col2:
                        st.metric("Clarity", f"{prod_scores.get('problem_clarity', 0)}/15")
                    with col3:
                        st.metric("Impact", f"{prod_scores.get('market_impact', 0)}/15")
                    with col4:
                        st.metric("Differentiation", f"{prod_scores.get('differentiation', 0)}/10")
                    
                    prod_reasoning = prod.get('product_reasoning', {})
                    
                    # Defensive: ensure reasoning is a dict
                    if not isinstance(prod_reasoning, dict):
                        prod_reasoning = {}
                    
                    if prod_reasoning.get('product_strengths'):
                        st.markdown("**✅ Product Strengths:**")
                        for s in prod_reasoning['product_strengths']:
                            st.markdown(f"- {s}")
                    
                    if prod_reasoning.get('product_gaps'):
                        st.markdown("**❌ Product Gaps:**")
                        for g in prod_reasoning['product_gaps']:
                            st.markdown(f"- {g}")
                    
                    if prod_reasoning.get('improvements'):
                        st.markdown("**🔨 Product Improvements:**")
                        for imp in prod_reasoning['improvements']:
                            st.markdown(f"- {imp}")
                
                # Execution Judge Details
                if 'execution' in judge_details:
                    exec_j = judge_details['execution']
                    st.markdown("### ⚙️ Execution & Feasibility Judge")
                    
                    exec_scores = exec_j.get('execution_scores', {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Feasibility", f"{exec_scores.get('implementation_feasibility', 0)}/15")
                    with col2:
                        st.metric("Demo Quality", f"{exec_scores.get('demo_quality', 0)}/10")
                    with col3:
                        st.metric("Deployment", f"{exec_scores.get('deployment_readiness', 0)}/10")
                    with col4:
                        st.metric("Completeness", f"{exec_scores.get('mvp_completeness', 0)}/5")
                    
                    exec_reasoning = exec_j.get('execution_reasoning', {})
                    
                    # Defensive: ensure reasoning is a dict
                    if not isinstance(exec_reasoning, dict):
                        exec_reasoning = {}
                    
                    if exec_reasoning.get('execution_strengths'):
                        st.markdown("**✅ Execution Strengths:**")
                        for s in exec_reasoning['execution_strengths']:
                            st.markdown(f"- {s}")
                    
                    if exec_reasoning.get('practical_concerns'):
                        st.markdown("**❌ Practical Concerns:**")
                        for c in exec_reasoning['practical_concerns']:
                            st.markdown(f"- {c}")
                    
                    if exec_reasoning.get('improvements'):
                        st.markdown("**🔨 Execution Improvements:**")
                        for imp in exec_reasoning['improvements']:
                            st.markdown(f"- {imp}")
        
        # Export
        st.markdown("### 📥 Export Report")
        output_path = os.path.join(OUTPUT_DIR, "final_results.xlsx")
        results_df.to_excel(output_path, index=False)
        with open(output_path, "rb") as f:
            st.download_button(
                "Download Excel Analysis", 
                f, 
                file_name="hackathon_ranking_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ============= DETAILED ARCHITECTURE REVIEW FEATURE =============
    st.markdown("---")
    st.markdown("### 🧠 Deep Architecture Analysis")
    st.info("Select a team to generate a comprehensive 'Senior Architect' critique of their system design.")
    
    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        selected_team = st.selectbox("Select Team:", df['Team Name'].tolist(), label_visibility="collapsed")
    
    with c_btn:
        generate_btn = st.button("Generate Report 🔬", use_container_width=True)
    
    if generate_btn:
        with st.status(f" Analyzing {selected_team}'s architecture...", expanded=True) as status:
            # 1. Download
            status.write("📥 Checking file...")
            row = df[df['Team Name'] == selected_team].iloc[0]
            download_dir = batch_download([row.to_dict()])
            local_path = download_dir.get(selected_team)
            
            if local_path and os.path.exists(local_path):
                # 2. Extract
                status.write("📄 Parsing diagrams & text...")
                content = process_file(local_path)
                
                if content:
                    # 3. Evaluate
                    status.write("🧠 Thinking (this may take 30-60s)...")
                    report = evaluate_architecture(content, selected_team, model_name=OLLAMA_MODEL)
                    
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    
                    st.subheader(f"Architecture Report: {selected_team}")
                    st.markdown(f"""
                        <div style="background-color: #222; padding: 20px; border-radius: 10px; font-family: 'Courier New', monospace;">
                            <pre style="white-space: pre-wrap;">{report}</pre>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button("📥 Download Text Report", report, file_name=f"{selected_team}_Architecture_Report.txt")
                    
                else:
                    status.update(label="❌ Extraction Failed", state="error")
                    st.error("Could not extract text content from the file.")
            else:
                status.update(label="❌ File Not Found", state="error")
                st.error("Could not download or find the project file.")
