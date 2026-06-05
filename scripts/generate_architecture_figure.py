import os
from pathlib import Path

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

import graphviz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "assets" / "dual_system_architecture"

# 创建有向图。
dot = graphviz.Digraph('Dual-System_Architecture', comment='Dual-System Cognitive Security Architecture')

# 设置全局图属性，使图片更适合论文展示。
dot.attr(rankdir='TB', size='12,12', dpi='300', fontname='Helvetica', splines='ortho')
dot.attr('node', shape='box', style='filled', fillcolor='white', fontname='Helvetica', fontsize='12')
dot.attr('edge', fontname='Helvetica', fontsize='10')

# --- 1. 输入空间 ---
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='Input Space: Adversarial Environment', style='dashed', color='#666666', bgcolor='#f9f9f9')
    c.node('User', 'User Query\n(Instruction)', shape='ellipse', fillcolor='#e1f5fe', color='#01579b')
    c.node('Attacker', 'Attacker\n(Indirect Injection)', shape='hexagon', fillcolor='#ffebee', color='#b71c1c')
    c.node('Web', 'External Knowledge Base\n(Web/Docs)', shape='cylinder', fillcolor='#eeeeee')
    c.node('Retrieval', 'Retrieved Context\n(Contains Poisoned Data)', style='filled,dashed', fillcolor='#fff3e0',
           color='#e65100')

    c.edge('User', 'Retrieval', label=' Search Query')
    c.edge('Attacker', 'Web', label=' Inject Malicious Prompt', style='dotted', color='red')
    c.edge('Web', 'Retrieval', label=' Fetch Content')

# --- 2. 核心防御架构 ---
with dot.subgraph(name='cluster_1') as c:
    c.attr(label='Proposed: Dual-System Cognitive Security Architecture', color='#1a237e', penwidth='2',
           bgcolor='#e8eaf6')

    # 系统一：快速感知层。
    with dot.subgraph(name='cluster_sys1') as s1:
        s1.attr(label='System 1: Intuitive Perception Layer (Fast Path)', color='#1565c0', style='filled',
                fillcolor='#bbdefb')
        s1.node('BERT', 'Fine-tuned DistilBERT\n(Text Classification)', shape='component')
        s1.node('PPL', 'Instruction Perplexity\n(Feature Extraction)')
        s1.node('Gate1', 'Confidence Gate\n(Thresholding)', shape='diamond', style='filled', fillcolor='#fff9c4')

    # 系统二：理性对齐层。
    with dot.subgraph(name='cluster_sys2') as s2:
        s2.attr(label='System 2: Rational Alignment Layer (Slow Path)', color='#2e7d32', style='filled',
                fillcolor='#c8e6c9')
        s2.node('Vector', 'Intent Consistency Check\n(Cosine Similarity in Embedding Space)', shape='box3d')
        s2.node('Auditor', 'Counterfactual Auditor Agent\n(Semantic Analysis)', shape='component')
        s2.node('Gate2', 'Safety Verdict', shape='diamond', style='filled', fillcolor='#fff9c4')

# --- 3. 执行空间 ---
with dot.subgraph(name='cluster_2') as c:
    c.attr(label='Execution Space', style='dashed', color='#666666')
    c.node('LLM', 'Victim LLM Agent\n(Search/Tool User)', shape='box', fillcolor='#e0f2f1')
    c.node('Safe', 'Safe Response\n(Task Aligned)', shape='note', fillcolor='#dcedc8')
    c.node('Block', 'Interception / Warning\n(Attack Detected)', shape='note', fillcolor='#ffcdd2')

# --- 模块连接关系 ---

# 输入进入系统一。
dot.edge('Retrieval', 'BERT', label=' Context Input')
dot.edge('Retrieval', 'PPL', label=' Text Features')
dot.edge('BERT', 'Gate1', label=' Risk Score')
dot.edge('PPL', 'Gate1')

# 第一层门控逻辑。
dot.edge('Gate1', 'LLM', label=' Low Risk\n(Pass)', color='green')
dot.edge('Gate1', 'Vector', label=' Ambiguous/High Risk\n(Activate System 2)', color='orange', penwidth='2')

# 第二层验证逻辑。
dot.edge('User', 'Vector', label=' Original Intent Embedding', style='dashed')
dot.edge('Vector', 'Gate2', label=' Vector Drift?')
dot.edge('Vector', 'Auditor', label=' Complex Logic Req.', style='dashed')
dot.edge('Auditor', 'Gate2', label=' Semantic Analysis')

# 第二层门控逻辑。
dot.edge('Gate2', 'LLM', label=' Alignment Verified', color='green')
dot.edge('Gate2', 'Block', label=' Injection Confirmed', color='red', penwidth='2')

# 大模型输出。
dot.edge('LLM', 'Safe', label=' Execute')

# 渲染到文档静态资源目录，避免输出到项目根目录。
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dot.render(str(OUTPUT_PATH), format='png', cleanup=True)

str(OUTPUT_PATH.with_suffix(".png"))
