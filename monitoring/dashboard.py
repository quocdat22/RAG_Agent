"""
Streamlit dashboard for monitoring RAG Agent metrics.
Run with: streamlit run monitoring/dashboard.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional

from monitoring.metrics_store import get_metrics_store


st.set_page_config(
    page_title="RAG Agent Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RAG Agent Monitoring Dashboard")

# Initialize metrics store
try:
    metrics_store = get_metrics_store()
except Exception as e:
    st.error(f"Không thể kết nối với metrics database: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
    index=1,
)

conversation_filter = st.sidebar.text_input("Conversation ID (optional)", "")

# Calculate time range
now = datetime.now()
if time_range == "Last 24 hours":
    start_time = (now - timedelta(days=1)).timestamp()
elif time_range == "Last 7 days":
    start_time = (now - timedelta(days=7)).timestamp()
elif time_range == "Last 30 days":
    start_time = (now - timedelta(days=30)).timestamp()
else:
    start_time = None

end_time = now.timestamp()

# Get metrics
try:
    # First, try to get all metrics without time filter to check if data exists
    all_metrics = metrics_store.get_query_metrics(limit=1)
    
    query_metrics = metrics_store.get_query_metrics(
        limit=1000,
        conversation_id=conversation_filter if conversation_filter else None,
        start_time=start_time,
        end_time=end_time,
    )
    latency_stats = metrics_store.get_latency_stats(
        start_time=start_time,
        end_time=end_time,
    )
    retrieval_stats = metrics_store.get_retrieval_stats(
        start_time=start_time,
        end_time=end_time,
    )
    chunk_stats = metrics_store.get_chunk_quality_stats(
        start_time=start_time,
        end_time=end_time,
    )
    
    # Debug info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Debug Info")
        st.write(f"Total metrics in DB: {len(all_metrics)}")
        st.write(f"Filtered metrics: {len(query_metrics)}")
        if all_metrics and not query_metrics:
            st.warning("⚠️ Có dữ liệu nhưng filter không match. Thử chọn 'All time'.")
except Exception as e:
    st.error(f"Lỗi khi lấy metrics: {e}")
    import traceback
    with st.expander("Chi tiết lỗi"):
        st.code(traceback.format_exc())
    st.stop()

# Overview metrics
st.header("📈 Overview Metrics")

if query_metrics:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_queries = len(query_metrics)
        st.metric("Total Queries", total_queries)
    
    with col2:
        avg_latency = latency_stats.get("avg_total_latency", 0)
        st.metric("Avg Total Latency", f"{avg_latency:.2f} ms" if avg_latency else "N/A")
    
    with col3:
        avg_retrieval = latency_stats.get("avg_retrieval_latency", 0)
        st.metric("Avg Retrieval Latency", f"{avg_retrieval:.2f} ms" if avg_retrieval else "N/A")
    
    with col4:
        avg_chunk_score = chunk_stats.get("avg_score", 0)
        st.metric("Avg Chunk Score", f"{avg_chunk_score:.3f}" if avg_chunk_score else "N/A")
else:
    st.info("Chưa có dữ liệu metrics. Hãy thực hiện một số queries để xem metrics.")

# Latency Analysis
st.header("⏱️ Query Latency Analysis")

if query_metrics:
    df = pd.DataFrame(query_metrics)
    df["created_at"] = pd.to_datetime(df["created_at"])
    
    # Latency over time
    if "total_latency_ms" in df.columns:
        fig = px.line(
            df,
            x="created_at",
            y="total_latency_ms",
            title="Total Query Latency Over Time",
            labels={"total_latency_ms": "Latency (ms)", "created_at": "Time"},
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        if "retrieval_latency_ms" in df.columns and "reranking_latency_ms" in df.columns and "llm_latency_ms" in df.columns:
            latency_breakdown = pd.DataFrame({
                "Phase": ["Retrieval", "Reranking", "LLM"],
                "Avg Latency (ms)": [
                    df["retrieval_latency_ms"].mean() if "retrieval_latency_ms" in df.columns else 0,
                    df["reranking_latency_ms"].mean() if "reranking_latency_ms" in df.columns else 0,
                    df["llm_latency_ms"].mean() if "llm_latency_ms" in df.columns else 0,
                ],
            })
            fig = px.bar(
                latency_breakdown,
                x="Phase",
                y="Avg Latency (ms)",
                title="Average Latency by Phase",
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "total_latency_ms" in df.columns:
            fig = px.histogram(
                df,
                x="total_latency_ms",
                title="Total Latency Distribution",
                labels={"total_latency_ms": "Latency (ms)", "count": "Frequency"},
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chưa có dữ liệu latency.")

# Retrieval Quality
st.header("🎯 Retrieval Quality")

if retrieval_stats:
    df_retrieval = pd.DataFrame(retrieval_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "avg_score" in df_retrieval.columns:
            fig = px.bar(
                df_retrieval,
                x="retrieval_method",
                y="avg_score",
                title="Average Score by Retrieval Method",
                labels={"avg_score": "Average Score", "retrieval_method": "Method"},
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "avg_num_results" in df_retrieval.columns:
            fig = px.bar(
                df_retrieval,
                x="retrieval_method",
                y="avg_num_results",
                title="Average Number of Results by Method",
                labels={"avg_num_results": "Avg Results", "retrieval_method": "Method"},
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chưa có dữ liệu retrieval quality.")

# Chunk Quality
st.header("📄 Chunk Quality Metrics")

if chunk_stats and chunk_stats.get("total_chunks", 0) > 0:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Chunks", chunk_stats.get("total_chunks", 0))
    
    with col2:
        st.metric("Final Chunks", chunk_stats.get("final_chunks", 0))
    
    with col3:
        avg_score = chunk_stats.get("avg_score", 0)
        st.metric("Average Score", f"{avg_score:.3f}" if avg_score else "N/A")
    
    # Score distribution (if we have individual chunk data)
    # Note: This would require querying chunk_quality table directly
    # For now, we show summary stats
    st.info("💡 Để xem phân phối score chi tiết, cần query trực tiếp từ chunk_quality table.")
else:
    st.info("Chưa có dữ liệu chunk quality.")

# Recent Queries
st.header("📋 Recent Queries")

if query_metrics:
    # Show last 20 queries
    recent_df = pd.DataFrame(query_metrics[:20])
    
    # Select columns to display
    display_cols = ["query", "total_latency_ms", "num_candidates", "num_final_chunks", "created_at"]
    available_cols = [col for col in display_cols if col in recent_df.columns]
    
    if available_cols:
        st.dataframe(
            recent_df[available_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Không có dữ liệu để hiển thị.")
else:
    st.info("Chưa có queries nào.")

# Footer
st.markdown("---")
st.markdown("**RAG Agent Monitoring Dashboard** | Dữ liệu được cập nhật theo thời gian thực")

