"""
Generate a comprehensive HTML technical report for the CRL-Atari project.

Produces a standalone HTML file with embedded CSS and optional MathJax,
covering architecture, training, normalization, consolidation methods,
evaluation pipeline, and data flow.

Usage:
    python scripts/generate_report.py [--output docs/CRL_Atari_Technical_Report.html]
"""

import argparse
import os
import sys
import textwrap
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Color scheme (matches visualize.py academic palette) ────────────
NAVY = "#2E5090"
TERRACOTTA = "#C45B28"
PLUM = "#6B4C9A"
CRIMSON = "#B5322A"
SAGE = "#5A8A3C"
SLATE = "#5B7B8A"
GOLD = "#C49A2A"
TEAL = "#2A7B7B"


def get_css() -> str:
    """Return the full embedded CSS for the report."""
    return f"""
    :root {{
        --navy: {NAVY};
        --terracotta: {TERRACOTTA};
        --plum: {PLUM};
        --crimson: {CRIMSON};
        --sage: {SAGE};
        --slate: {SLATE};
        --gold: {GOLD};
        --teal: {TEAL};
        --bg: #FAFAF8;
        --text: #2C2C2C;
        --text-light: #555555;
        --border: #D8D6D2;
        --code-bg: #F3F2EE;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    html {{
        scroll-behavior: smooth;
        font-size: 16px;
    }}

    body {{
        font-family: "Georgia", "Times New Roman", "DejaVu Serif", serif;
        color: var(--text);
        background: var(--bg);
        line-height: 1.72;
        max-width: 52rem;
        margin: 0 auto;
        padding: 0 2rem 6rem;
    }}

    /* ── Title page ────────────────────────────────────────── */
    .title-page {{
        text-align: center;
        padding: 6rem 0 4rem;
        border-bottom: 2px solid var(--navy);
        margin-bottom: 3rem;
    }}
    .title-page h1 {{
        font-size: 2.8rem;
        color: var(--navy);
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }}
    .title-page .subtitle {{
        font-size: 1.15rem;
        color: var(--text-light);
        font-style: italic;
        margin-bottom: 1.5rem;
    }}
    .title-page .methods {{
        font-size: 1rem;
        color: var(--slate);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }}
    .title-page .author {{
        font-size: 1.05rem;
        color: var(--text);
        margin-bottom: 0.3rem;
    }}
    .title-page .date {{
        font-size: 0.95rem;
        color: var(--text-light);
    }}

    /* ── Table of Contents ──────────────────────────────────── */
    nav.toc {{
        background: white;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 3rem;
    }}
    nav.toc h2 {{
        font-size: 1.3rem;
        color: var(--navy);
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
    }}
    nav.toc ol {{
        padding-left: 1.6rem;
    }}
    nav.toc li {{
        margin-bottom: 0.35rem;
    }}
    nav.toc a {{
        color: var(--navy);
        text-decoration: none;
        transition: color 0.2s;
    }}
    nav.toc a:hover {{
        color: var(--terracotta);
        text-decoration: underline;
    }}

    /* ── Sections ──────────────────────────────────────────── */
    section {{
        margin-bottom: 3rem;
    }}
    section h2 {{
        font-size: 1.55rem;
        color: var(--navy);
        border-bottom: 2px solid var(--navy);
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
    }}
    section h3 {{
        font-size: 1.15rem;
        color: var(--plum);
        margin: 1.5rem 0 0.6rem;
    }}
    section h4 {{
        font-size: 1.0rem;
        color: var(--terracotta);
        margin: 1.2rem 0 0.4rem;
    }}
    p {{
        margin-bottom: 0.9rem;
        text-align: justify;
    }}

    /* ── Code blocks ───────────────────────────────────────── */
    pre {{
        background: var(--code-bg);
        border: 1px solid var(--border);
        border-left: 4px solid var(--navy);
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin: 1rem 0 1.2rem;
        overflow-x: auto;
        font-family: "SF Mono", "Fira Code", "Consolas", monospace;
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--text);
    }}
    code {{
        font-family: "SF Mono", "Fira Code", "Consolas", monospace;
        font-size: 0.88em;
        background: var(--code-bg);
        padding: 0.15em 0.4em;
        border-radius: 3px;
    }}
    pre code {{
        background: none;
        padding: 0;
    }}

    /* ── Tables ────────────────────────────────────────────── */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1.2rem 0;
        font-size: 0.92rem;
    }}
    th, td {{
        padding: 0.55rem 0.8rem;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }}
    th {{
        background: var(--navy);
        color: white;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
    }}
    tr:nth-child(even) td {{
        background: rgba(46, 80, 144, 0.04);
    }}
    tr:hover td {{
        background: rgba(46, 80, 144, 0.08);
    }}
    td.category {{
        font-weight: bold;
        color: white;
        font-size: 0.82rem;
        letter-spacing: 0.03em;
    }}
    td.cat-model {{ background: {NAVY}; }}
    td.cat-training {{ background: {TERRACOTTA}; }}
    td.cat-norm {{ background: {SAGE}; }}
    td.cat-ewc {{ background: {GOLD}; }}
    td.cat-distill {{ background: {PLUM}; }}
    td.cat-htcl {{ background: {CRIMSON}; }}
    td.cat-eval {{ background: {TEAL}; }}

    /* ── Flowchart ─────────────────────────────────────────── */
    .flowchart {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0;
    }}
    .flow-row {{
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        width: 100%;
    }}
    .flow-box {{
        border: 2px solid var(--navy);
        border-radius: 8px;
        padding: 0.7rem 1.2rem;
        text-align: center;
        font-size: 0.88rem;
        font-weight: bold;
        min-width: 140px;
        transition: transform 0.15s;
    }}
    .flow-box:hover {{
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
    }}
    .flow-box small {{
        display: block;
        font-weight: normal;
        font-style: italic;
        font-size: 0.78rem;
        color: var(--text-light);
        margin-top: 0.2rem;
    }}
    .flow-arrow {{
        font-size: 1.5rem;
        color: var(--navy);
        text-align: center;
    }}
    .flow-label {{
        font-size: 0.8rem;
        color: var(--text-light);
        font-style: italic;
        text-align: center;
    }}

    .bg-blue {{ background: rgba(46,80,144,0.12); border-color: {NAVY}; }}
    .bg-orange {{ background: rgba(196,91,40,0.12); border-color: {TERRACOTTA}; }}
    .bg-purple {{ background: rgba(107,76,154,0.12); border-color: {PLUM}; }}
    .bg-red {{ background: rgba(181,50,42,0.12); border-color: {CRIMSON}; }}
    .bg-green {{ background: rgba(90,138,60,0.12); border-color: {SAGE}; }}
    .bg-gold {{ background: rgba(196,154,42,0.12); border-color: {GOLD}; }}
    .bg-teal {{ background: rgba(42,123,123,0.12); border-color: {TEAL}; }}

    /* ── Callout boxes ─────────────────────────────────────── */
    .callout {{
        border-left: 4px solid var(--terracotta);
        background: rgba(196,91,40,0.06);
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.93rem;
    }}
    .callout.info {{
        border-left-color: var(--navy);
        background: rgba(46,80,144,0.06);
    }}
    .callout.success {{
        border-left-color: var(--sage);
        background: rgba(90,138,60,0.06);
    }}

    /* ── Equation blocks ───────────────────────────────────── */
    .equation {{
        background: white;
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
        text-align: center;
        font-family: "Georgia", serif;
        font-style: italic;
        font-size: 0.95rem;
        overflow-x: auto;
    }}

    /* ── Lists ─────────────────────────────────────────────── */
    ul, ol {{
        margin: 0.5rem 0 1rem 1.8rem;
    }}
    li {{
        margin-bottom: 0.3rem;
    }}

    /* ── Footer ────────────────────────────────────────────── */
    footer {{
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        text-align: center;
        font-size: 0.85rem;
        color: var(--text-light);
    }}

    /* ── Print styles ──────────────────────────────────────── */
    @media print {{
        body {{ max-width: none; padding: 0 1cm; }}
        .flow-box:hover {{ transform: none; box-shadow: none; }}
        section {{ page-break-inside: avoid; }}
    }}
"""


def generate_report(output_path: str = "docs/CRL_Atari_Technical_Report.html") -> None:
    """Generate a standalone HTML technical report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRL-Atari Technical Report</title>
    <style>{get_css()}</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════════
     TITLE PAGE
     ═══════════════════════════════════════════════════════════ -->
<div class="title-page">
    <h1>CRL-Atari</h1>
    <div class="subtitle">
        Continual Reinforcement Learning on Atari Games<br>
        Technical Report &mdash; Architecture, Methods, and Pipeline
    </div>
    <div class="methods">Knowledge Distillation &nbsp;|&nbsp; HTCL</div>
    <div class="author">Protik Nag</div>
    <div class="date">{datetime.now().strftime("%B %Y")}</div>
</div>

<!-- ═══════════════════════════════════════════════════════════
     TABLE OF CONTENTS
     ═══════════════════════════════════════════════════════════ -->
<nav class="toc">
    <h2>Table of Contents</h2>
    <ol>
        <li><a href="#sec-overview">Overview and Motivation</a></li>
        <li><a href="#sec-architecture">System Architecture (Pipeline Flowchart)</a></li>
        <li><a href="#sec-dqn">DQN Network Architecture</a></li>
        <li><a href="#sec-action">Union Action Space and Action Masking</a></li>
        <li><a href="#sec-preproc">Atari Environment Preprocessing</a></li>
        <li><a href="#sec-training">DQN Agent and Training Loop</a></li>
        <li><a href="#sec-norm">Q-Value Normalization</a></li>
        <li><a href="#sec-init">Expert Initialization Strategy</a></li>
        <li><a href="#sec-distill">Knowledge Distillation Consolidation</a></li>
        <li><a href="#sec-htcl">HTCL Consolidation (Taylor Expansion)</a></li>
        <li><a href="#sec-lambda">Lambda Constraint and Catch-Up Iterations</a></li>
        <li><a href="#sec-eval">Evaluation and Comparison Pipeline</a></li>
        <li><a href="#sec-hyperparams">Summary of Hyperparameters</a></li>
    </ol>
</nav>

<!-- ═══════════════════════════════════════════════════════════
     1. OVERVIEW
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-overview">
    <h2>1. Overview and Motivation</h2>

    <p>This project implements a Continual Reinforcement Learning (CRL) framework for Atari games.
    The core challenge is training a single model that performs well across multiple Atari games
    without <strong>catastrophic forgetting</strong>, where learning a new task degrades performance
    on previously learned tasks.</p>

    <h3>Approach</h3>
    <p>The pipeline follows a <em>train experts, then consolidate</em> paradigm:</p>
    <ol>
        <li><strong>Expert Training</strong> &mdash; Train separate DQN agents, one per Atari game
            (Pong, Breakout, SpaceInvaders). All experts share a union action space
            (the union of each game&rsquo;s minimal actions, computed at runtime), with per-game action masking.</li>
        <li><strong>Consolidation</strong> &mdash; Merge the expert models into a single global
            model using one of three methods:
            <ul>
                <li><strong>Knowledge Distillation</strong> &mdash; Temperature-scaled soft-target matching</li>
                <li><strong>HTCL</strong> &mdash; Second-order Taylor expansion with closed-form parameter updates</li>
            </ul>
        </li>
        <li><strong>Evaluation</strong> &mdash; Compare the consolidated models against individual experts
            on all tasks to measure knowledge retention and forward/backward transfer.</li>
    </ol>

    <h3>Key Design Decisions</h3>
    <ul>
        <li><strong>Union action space</strong> &mdash; The output head covers the union of each game&rsquo;s
            minimal action set (6 actions for Pong/Breakout/SpaceInvaders). Invalid actions are
            masked to &minus;&infin;. This allows direct weight sharing and merging.</li>
        <li><strong>Independent expert initialization</strong> &mdash; Each expert is initialized from random
            weights, avoiding any coupling between the sequential training order.</li>
        <li><strong>PopArt normalization</strong> &mdash; Q-value output layer weights are rescaled on-the-fly to
            handle scale differences across games (Pong rewards in [&minus;1, 1] vs. SpaceInvaders in [0, 500+]).</li>
        <li><strong>Diagonal Fisher</strong> &mdash; Used by HTCL as a computationally tractable
            approximation to the Hessian, enabling element-wise closed-form updates.</li>
    </ul>

    <div class="callout info">
        <strong>Reference:</strong> Nag, P., Raghavan, K., Narayanan, V. (2026).
        &ldquo;Mitigating Task-Order Sensitivity and Forgetting via Hierarchical Second-Order Consolidation.&rdquo;
        ICML 2026. arXiv:2602.02568.
    </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     2. SYSTEM ARCHITECTURE FLOWCHART
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-architecture">
    <h2>2. System Architecture (Pipeline Flowchart)</h2>

    <div class="flowchart">
        <!-- Row 1: Config + Seed -->
        <div class="flow-row">
            <div class="flow-box bg-gold">Config<small>base.yaml</small></div>
            <div class="flow-box bg-green">Seed<small>Reproducibility</small></div>
        </div>

        <div class="flow-arrow">&darr;</div>
        <div class="flow-label">Independent Random Initialization per Expert</div>

        <!-- Row 2: Expert Training -->
        <div class="flow-row">
            <div class="flow-box bg-orange">Expert 1: Pong<small>DQN Training</small></div>
            <div class="flow-box bg-orange">Expert 2: Breakout<small>DQN Training</small></div>
            <div class="flow-box bg-orange">Expert 3: SpaceInvaders<small>DQN Training</small></div>
        </div>

        <div class="flow-arrow">&darr;</div>
        <div class="flow-label">Expert checkpoints + replay buffers</div>

        <!-- Row 3: Consolidation Methods -->
        <div class="flow-row">
            <div class="flow-box bg-purple">Distillation<small>KL-div soft targets</small></div>
            <div class="flow-box bg-red">HTCL<small>Taylor 2nd-order</small></div>
        </div>

        <div class="flow-arrow">&darr;</div>

        <!-- Row 5: Consolidated Models -->
        <div class="flow-row">
            <div class="flow-box bg-green">Distill Model<small>consolidated_distill.pt</small></div>
            <div class="flow-box bg-green">HTCL Model<small>consolidated_htcl.pt</small></div>
        </div>

        <div class="flow-arrow">&darr;</div>

        <!-- Row 6: Evaluation -->
        <div class="flow-row">
            <div class="flow-box bg-blue" style="min-width: 360px; font-size: 1rem;">
                Evaluation on All Tasks
                <small>Expert vs Distillation vs HTCL</small>
            </div>
        </div>

        <div class="flow-arrow">&darr;</div>

        <!-- Row 7: Outputs -->
        <div class="flow-row">
            <div class="flow-box bg-blue">Comparison Plots<small>PNG + SVG</small></div>
            <div class="flow-box bg-blue">Heatmap<small>Performance matrix</small></div>
            <div class="flow-box bg-blue">JSON Results<small>Machine-readable</small></div>
        </div>
    </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     3. DQN NETWORK ARCHITECTURE
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-dqn">
    <h2>3. DQN Network Architecture</h2>

    <p>The DQN uses a standard convolutional neural network (CNN) backbone followed by fully
    connected layers. The key architectural choice is the <strong>union action space output head</strong>
    that covers the union of minimal action sets across all games.</p>

    <h3>Architecture</h3>
<pre><code>Input:  (batch, 4, 84, 84)    # 4 stacked grayscale frames, 84&times;84 pixels

Layer 1: Conv2d(4 &rarr; 32,  kernel=8&times;8, stride=4, padding=0) + ReLU
         Output: (batch, 32, 20, 20)

Layer 2: Conv2d(32 &rarr; 64, kernel=4&times;4, stride=2, padding=0) + ReLU
         Output: (batch, 64, 9, 9)

Layer 3: Conv2d(64 &rarr; 64, kernel=3&times;3, stride=1, padding=0) + ReLU
         Output: (batch, 64, 7, 7)

Flatten: (batch, 64 &times; 7 &times; 7) = (batch, 3136)

FC:      Linear(3136 &rarr; 512) + ReLU

Output:  Linear(512 &rarr; 6)   # Q-values for union action space</code></pre>

    <p><strong>Total parameters:</strong> ~1.7 million</p>

    <h3>Dueling DQN (optional)</h3>
    <p>Instead of a single output head, splits into two streams:</p>
    <ul>
        <li><strong>Value stream:</strong> Linear(512 &rarr; 256) &rarr; Linear(256 &rarr; 1)</li>
        <li><strong>Advantage stream:</strong> Linear(512 &rarr; 256) &rarr; Linear(256 &rarr; 6)</li>
    </ul>
    <div class="equation">
        Q(s, a) = V(s) + A(s, a) &minus; mean<sub>a'</sub> A(s, a')
    </div>

    <h3>Why a Union Action Space?</h3>
    <p>Rather than using all 18 ALE joystick actions, we compute the <em>union</em> of each game&rsquo;s
    minimal action set at runtime. For Pong + Breakout + SpaceInvaders this yields 6 actions:</p>
    <table>
        <tr><th>Game</th><th>Minimal Actions</th><th>ALE Indices</th></tr>
        <tr><td>Pong</td><td>6 actions</td><td>&#123;0, 1, 3, 4, 11, 12&#125;</td></tr>
        <tr><td>Breakout</td><td>4 actions</td><td>&#123;0, 1, 3, 4&#125;</td></tr>
        <tr><td>SpaceInvaders</td><td>6 actions</td><td>&#123;0, 1, 3, 4, 11, 12&#125;</td></tr>
    </table>
    <p><strong>Union = &#123;0, 1, 3, 4, 11, 12&#125;</strong> = NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE (6 actions).
    This is a much smaller head than the full 18-action space, resulting in fewer parameters
    and a more efficient representation. Invalid actions per game are still masked to &minus;&infin;.</p>
</section>

<!-- ═══════════════════════════════════════════════════════════
     4. UNIFIED ACTION SPACE
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-action">
    <h2>4. Union Action Space and Action Masking</h2>

    <p>The Arcade Learning Environment (ALE) defines 18 possible actions for the Atari joystick:</p>

    <table>
        <tr><th>Index</th><th>Action</th><th>Index</th><th>Action</th><th>Index</th><th>Action</th></tr>
        <tr><td>0</td><td>NOOP</td><td>6</td><td>UPRIGHT</td><td>12</td><td>LEFTFIRE</td></tr>
        <tr><td>1</td><td>FIRE</td><td>7</td><td>UPLEFT</td><td>13</td><td>DOWNFIRE</td></tr>
        <tr><td>2</td><td>UP</td><td>8</td><td>DOWNRIGHT</td><td>14</td><td>UPRIGHTFIRE</td></tr>
        <tr><td>3</td><td>RIGHT</td><td>9</td><td>DOWNLEFT</td><td>15</td><td>UPLEFTFIRE</td></tr>
        <tr><td>4</td><td>LEFT</td><td>10</td><td>UPFIRE</td><td>16</td><td>DOWNRIGHTFIRE</td></tr>
        <tr><td>5</td><td>DOWN</td><td>11</td><td>RIGHTFIRE</td><td>17</td><td>DOWNLEFTFIRE</td></tr>
    </table>

    <h3>Action Masking</h3>
    <p>During action selection, the agent computes Q(s, a) for all 6 union actions, then applies:</p>
    <div class="equation">
        mask[a] = 0 if a &in; valid_actions, &minus;&infin; otherwise<br><br>
        Q<sub>masked</sub>(s, a) = Q(s, a) + mask[a]<br><br>
        a* = argmax<sub>a</sub> Q<sub>masked</sub>(s, a)
    </div>
    <p>This ensures the agent <strong>never</strong> selects an invalid action while maintaining a
    shared weight structure.</p>

    <h3>Action Mapping (UnionActionWrapper)</h3>
    <p>The ALE environment expects <em>local</em> action indices (0 to n_game_actions &minus; 1),
    not union indices. The <code>UnionActionWrapper</code> translates:</p>
<pre><code>Example for Pong (minimal actions = [0, 1, 3, 4, 11, 12]):
    Union = [0, 1, 3, 4, 11, 12]  (all 6 are valid for Pong)
    union index 0 (NOOP)       &rarr; local index 0
    union index 1 (FIRE)       &rarr; local index 1
    union index 2 (RIGHT)      &rarr; local index 2
    union index 3 (LEFT)       &rarr; local index 3
    union index 4 (RIGHTFIRE)  &rarr; local index 4
    union index 5 (LEFTFIRE)   &rarr; local index 5

Example for Breakout (minimal actions = [0, 1, 3, 4]):
    union index 0 (NOOP)       &rarr; local index 0
    union index 1 (FIRE)       &rarr; local index 1
    union index 2 (RIGHT)      &rarr; local index 2
    union index 3 (LEFT)       &rarr; local index 3
    union index 4 (RIGHTFIRE)  &rarr; INVALID (masked to &minus;&infin;)
    union index 5 (LEFTFIRE)   &rarr; INVALID (masked to &minus;&infin;)</code></pre>
    <p>This wrapper is the <strong>outermost</strong> wrapper (applied last, after all other wrappers)
    so that inner wrappers such as <code>FireResetEnv</code> still operate on the game&rsquo;s local action
    space. The agent issues union indices, which the wrapper maps to local indices before
    passing them down.</p>

    <h3>Epsilon-Greedy Exploration</h3>
    <p>During training, with probability &epsilon;, the agent selects a random action
    uniformly from the valid action set (not all 6 union actions):</p>
    <div class="equation">
        a ~ Uniform(valid_actions) with probability &epsilon;<br>
        a = argmax Q<sub>masked</sub>(s, a) with probability 1 &minus; &epsilon;
    </div>
    <p>Epsilon decays linearly from 1.0 to 0.01 over 250,000 steps.</p>
</section>

<!-- ═══════════════════════════════════════════════════════════
     5. ATARI PREPROCESSING
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-preproc">
    <h2>5. Atari Environment Preprocessing</h2>

    <h3>Environment Wrapper Stack (DeepMind-style)</h3>
    <p>The Atari environment goes through a chain of wrappers that preprocess frames and modify
    the reward/termination signals. Applied in this exact order:</p>
    <ol>
        <li><code>gym.make(env_id)</code> &mdash; Raw ALE environment</li>
        <li><code>NoopResetEnv(noop_max=30)</code> &mdash; Random 1&ndash;30 NOOP actions on reset</li>
        <li><code>MaxAndSkipEnv(skip=4)</code> &mdash; Repeat action 4&times;, return max of last 2 frames</li>
        <li><code>EpisodicLifeEnv</code> &mdash; Treat life loss as episode end (training only)</li>
        <li><code>FireResetEnv</code> &mdash; Press FIRE after reset (if game requires it)</li>
        <li><code>WarpFrame(84, 84)</code> &mdash; RGB &rarr; grayscale, resize to 84&times;84</li>
        <li><code>ClipRewardEnv</code> &mdash; Clip rewards to &#123;&minus;1, 0, +1&#125; via <code>np.sign</code> (training only)</li>
        <li><code>FrameStack(k=4)</code> &mdash; Stack last 4 frames &rarr; (4, 84, 84)</li>
        <li><code>UnionActionWrapper</code> &mdash; Union &rarr; local action mapping (<strong>outermost</strong>)</li>
    </ol>

    <div class="callout">
        <strong>Why this order matters:</strong> <code>UnionActionWrapper</code> must be the <strong>outermost</strong>
        wrapper so that inner wrappers like <code>FireResetEnv</code> still see the game&rsquo;s native local
        action space. The agent issues union action indices, which the outermost wrapper maps to
        local indices before passing down through the wrapper chain. <code>EpisodicLifeEnv</code>
        and <code>ClipRewardEnv</code> are used only during training. Evaluation uses true game-over
        and raw unclipped rewards.
    </div>

    <h3>Observations</h3>
    <p>Shape: <code>(4, 84, 84)</code>, dtype: <code>uint8</code>, range: [0, 255] (converted to
    [0, 1] float by the agent before the network forward pass).</p>

    <h3>Replay Buffer</h3>
    <p>Stores transitions as uint8 numpy arrays (memory-efficient):
    <code>(state, action, reward, next_state, done)</code> with state shape (4, 84, 84).
    Capacity: 200,000 transitions (default) or 1,000 (debug mode). Sampling: random uniform.
    Pixels normalized to [0, 1] float on sample.</p>
</section>

<!-- ═══════════════════════════════════════════════════════════
     6. DQN TRAINING LOOP
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-training">
    <h2>6. DQN Agent and Training Loop</h2>

    <p>For each expert (one per Atari game), we run the standard DQN algorithm with the
    following components:</p>

    <h3>Double Network Architecture</h3>
    <ul>
        <li><strong>Policy network</strong> &mdash; Updated every <code>train_freq=4</code> environment steps</li>
        <li><strong>Target network</strong> &mdash; Hard-copied from policy every <code>target_update_freq=2500</code> steps</li>
    </ul>

    <h3>Target Q-Value Computation</h3>
    <p>For a transition (s, a, r, s', done):</p>
    <div class="equation">
        y = r + &gamma; &middot; max<sub>a' &in; valid</sub> Q<sub>target</sub>(s', a') &middot; (1 &minus; done)
    </div>
    <p>where &gamma; = 0.99 and the max is taken over valid actions only (masked).</p>

    <h3>Loss Function</h3>
    <p>Smooth L1 (Huber) loss between predicted and target Q-values:</p>
    <div class="equation">L = SmoothL1(Q<sub>policy</sub>(s, a), y)</div>
    <p>This is more robust to outliers than MSE.</p>

    <h3>Training Schedule</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Total timesteps</td><td>1,000,000 per expert (5,000 debug)</td></tr>
        <tr><td>Replay buffer min fill</td><td>10,000 transitions</td></tr>
        <tr><td>Training frequency</td><td>Every 4 environment steps</td></tr>
        <tr><td>Epsilon decay</td><td>1.0 &rarr; 0.01 linearly over 250K steps</td></tr>
        <tr><td>Gradient clipping</td><td>max_norm = 10.0</td></tr>
        <tr><td>Optimizer</td><td>AdamW (lr = 1e-4)</td></tr>
    </table>

    <h3>Checkpointing</h3>
    <ul>
        <li>Best model (by evaluation reward) saved automatically as <code>expert_{Game}_best.pt</code></li>
        <li>Periodic step checkpoints are deleted when a new best is found (keeps only the best)</li>
        <li>Each checkpoint contains: policy_net, target_net, optimizer, normalizer, step count</li>
    </ul>
</section>

<!-- ═══════════════════════════════════════════════════════════
     7. Q-VALUE NORMALIZATION
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-norm">
    <h2>7. Q-Value Normalization</h2>

    <p>Different Atari games have vastly different reward scales:</p>
    <table>
        <tr><th>Game</th><th>Reward Range</th></tr>
        <tr><td>Pong</td><td>&#123;&minus;1, 0, +1&#125;</td></tr>
        <tr><td>Breakout</td><td>&#123;0, 1, 4, 7&#125; (per brick)</td></tr>
        <tr><td>SpaceInvaders</td><td>&#123;0, 5, 10, 15, 20, 25, 30, &hellip;&#125;</td></tr>
    </table>
    <p>Without normalization, high-reward games would dominate the consolidation process.</p>

    <h3>Method: PopArt Normalization (Default)</h3>
    <p><em>&ldquo;Preserving Outputs Precisely while Adaptively Rescaling Targets&rdquo;</em>
    (van Hasselt et al., 2016). Instead of normalizing Q-values externally, PopArt modifies
    the output layer weights to preserve the network&rsquo;s predictions while adapting to
    changing target statistics:</p>
    <div class="equation">
        &mu;(t) = (1 &minus; m) &middot; &mu;(t&minus;1) + m &middot; mean(targets)<br>
        &sigma;(t) = (1 &minus; m) &middot; &sigma;(t&minus;1) + m &middot; var(targets)
    </div>
    <p>When the running statistics change, the output layer weights and biases are rescaled
    to preserve the network&rsquo;s unnormalized outputs:</p>
    <div class="equation">
        W<sub>new</sub> = W<sub>old</sub> &middot; &sigma;<sub>old</sub> / &sigma;<sub>new</sub><br>
        b<sub>new</sub> = (&sigma;<sub>old</sub> &middot; b<sub>old</sub> + &mu;<sub>old</sub> &minus; &mu;<sub>new</sub>) / &sigma;<sub>new</sub>
    </div>
    <p>Parameters: momentum m = 0.01.</p>

    <h4>Integration with Double DQN</h4>
    <p>During training, the target network&rsquo;s outputs are <em>denormalized</em> before computing
    the Bellman target (r + &gamma; max Q<sub>target</sub>). The resulting target is then
    <em>normalized</em> before computing the loss against the policy network&rsquo;s output. This
    ensures the loss is computed in the normalized space while the Bellman backup operates
    in the original reward scale. Action selection via argmax is invariant to affine transforms.</p>

    <div class="callout success">
        <strong>Significance for consolidation:</strong> PopArt handles reward-scale differences
        between games naturally&mdash;no external normalization needed. In knowledge distillation,
        Q-values are additionally normalized per task before computing KL divergence.
    </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     8. EXPERT INITIALIZATION STRATEGY
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-init">
    <h2>8. Expert Initialization Strategy</h2>

    <p>Each expert is initialized independently with <strong>random weights</strong>. There is no
    shared global model or weight averaging between experts during training.</p>

    <h3>Procedure</h3>
    <ol>
        <li><strong>Expert 1 (Pong):</strong> Random init &rarr; Train on Pong for T steps &rarr; Save best checkpoint</li>
        <li><strong>Expert 2 (Breakout):</strong> Random init &rarr; Train on Breakout for T steps &rarr; Save best checkpoint</li>
        <li><strong>Expert 3 (SpaceInvaders):</strong> Random init &rarr; Train on SpaceInvaders for T steps &rarr; Save best checkpoint</li>
    </ol>

    <h3>Rationale</h3>
    <ul>
        <li>Independent initialization avoids coupling between the sequential training order&mdash;the
            order of games does not affect individual expert quality.</li>
        <li>Each expert is free to find the best representation for its specific game without
            being biased by features from other games.</li>
        <li>The consolidation methods (Distillation, HTCL) are specifically designed to
            merge independently trained experts. They do not require experts to start from a
            common initialization.</li>
        <li>For HTCL, the initial global model for consolidation is simply the parameter average
            of all experts: w<sub>global</sub> = (1/T) &middot; &Sigma; w<sub>expert_t</sub>.
            The Taylor expansion operates from this averaged starting point.</li>
    </ul>

    <div class="callout info">
        <strong>Note:</strong> Previous versions of this pipeline used global-to-local initialization
        (initializing each expert from a shared running average). This was removed because it
        coupled expert quality to training order and did not provide consistent benefits for
        the consolidation methods used here.
    </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     9. KNOWLEDGE DISTILLATION
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-distill">
    <h2>9. Knowledge Distillation Consolidation</h2>

    <h3>Concept</h3>
    <p>Train a &ldquo;student&rdquo; model (the global/consolidated model) to mimic the output
    distributions of multiple &ldquo;teacher&rdquo; models (the experts). Uses temperature-scaled
    softmax and KL divergence as the matching criterion.</p>

    <h3>Step 1: Collect Q-Value Statistics</h3>
    <p>For each expert/task, sample states from the replay buffer and compute
    &mu;<sub>task</sub> (mean) and &sigma;<sub>task</sub> (std) of Q-values. These are
    used for per-task normalization.</p>

    <h3>Step 2: Distillation Loss (Per Task)</h3>
    <p>Given a batch of states from task <em>t</em>'s replay buffer:</p>
<pre><code>Q_teacher = frozen expert_t(states)  # shape: (batch, 6)
Q_student = global_model(states)     # shape: (batch, 6)</code></pre>
    <p>Normalize per task and compute soft targets:</p>
    <div class="equation">
        p<sub>teacher</sub> = softmax(Q&#770;<sub>teacher</sub> / &tau;) &nbsp;&nbsp;&nbsp;
        log q<sub>student</sub> = log_softmax(Q&#770;<sub>student</sub> / &tau;)
    </div>
    <p>KL divergence with standard Hinton scaling (&tau; = 2.0):</p>
    <div class="equation">
        L<sub>t</sub> = &tau;&sup2; &middot; KL(p<sub>teacher</sub> || q<sub>student</sub>)
    </div>

    <h3>Step 3: Training</h3>
    <p>For <code>distill_epochs=50</code> (5 in debug), iterate over tasks in random order,
    sample batch from each task's replay buffer, compute distillation loss, and update
    the student with AdamW (lr = 5e-5).</p>

    <h3>Advantages</h3>
    <ul>
        <li>Directly optimizes output similarity (behavioral cloning in Q-space).</li>
        <li>Does not require Fisher computation.</li>
        <li>Can capture richer information through the full Q-value distribution.</li>
    </ul>
</section>

<!-- ═══════════════════════════════════════════════════════════
     11. HTCL
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-htcl">
    <h2>11. HTCL Consolidation (Taylor Expansion)</h2>
    <p><em>Hierarchical Taylor Consolidation Learning</em>
    (Nag, Raghavan, Narayanan, 2026; arXiv:2602.02568)</p>

    <h3>Concept</h3>
    <p>Uses a second-order Taylor expansion of the loss function around the current global
    parameters. With a diagonal Fisher as the Hessian approximation, this yields a
    <strong>closed-form parameter update</strong> (no iterative optimization needed).</p>

    <h3>The Taylor Update</h3>
    <p>For each task <em>t</em>, with local expert weights w<sub>t</sub> and global model w<sub>global</sub>:</p>
    <div class="equation">
        w<sub>global</sub><sup>new</sup> = w<sub>global</sub> + &eta; &middot;
        (H + &lambda; I)<sup>&minus;1</sup> [&lambda; &middot; &delta;<sub>d</sub> &minus; g]
    </div>
    <p>where:</p>
    <ul>
        <li>&delta;<sub>d</sub> = w<sub>t</sub> &minus; w<sub>global</sub> (expert drift from global)</li>
        <li>g = &nabla;J(w<sub>global</sub>) (gradient of cumulative past loss at global)</li>
        <li>H = diag(F) (diagonal Fisher / Hessian approximation)</li>
        <li>&lambda; = regularization strength, &eta; = 0.9 (step size)</li>
    </ul>

    <h3>Diagonal Simplification</h3>
    <p>Since H is diagonal, the inverse is trivial (element-wise):</p>
    <div class="equation">
        (H + &lambda; I)<sup>&minus;1</sup><sub>ii</sub> = 1 / (H<sub>ii</sub> + &lambda;)
    </div>
    <p>So the update for each parameter <em>i</em> is:</p>
    <div class="equation">
        w<sub>i</sub><sup>new</sup> = w<sub>i</sub> + &eta; &middot;
        [&lambda; &middot; (w<sub>t,i</sub> &minus; w<sub>i</sub>) &minus; g<sub>i</sub>] / (H<sub>ii</sub> + &lambda;)
    </div>

    <h3>Sequential Task Consolidation</h3>
    <p>Tasks are merged one at a time into the global model:</p>
    <ol>
        <li>Merge task 1 (Pong) into global &rarr; updated global</li>
        <li>Merge task 2 (Breakout) into updated global &rarr; updated global</li>
        <li>Merge task 3 (SpaceInvaders) &rarr; final global</li>
    </ol>
    <p>After each merge, <strong>catch-up iterations</strong> refine the result (see next section).</p>

    <div class="callout info">
        <strong>Key insight:</strong> HTCL uses the
        full second-order expansion, yielding an optimal direction, not just regularization.
        HTCL's closed-form update is a single step (no iterative optimization). HTCL explicitly
        accounts for the gradient at the global position, not just the distance to task optima.
    </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     12. LAMBDA CONSTRAINT + CATCH-UP
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-lambda">
    <h2>12. Lambda Constraint and Catch-Up Iterations</h2>

    <h3>Lambda Constraint</h3>
    <p>The Taylor update requires (H + &lambda; I) to be positive definite:</p>
    <div class="equation">
        &lambda; > &minus;&mu;<sub>min</sub>(H)
    </div>
    <p>where &mu;<sub>min</sub>(H) is the minimum eigenvalue of the diagonal Hessian approximation.
    For a diagonal matrix, this equals the minimum diagonal entry: &mu;<sub>min</sub> = min<sub>i</sub>(F<sub>i</sub>).
    Since Fisher values are always &ge; 0, we need &lambda; > 0 (trivially satisfied for
    default &lambda; = 1.0).</p>

    <h4>Auto-Adjustment (<code>lambda_auto=true</code>)</h4>
    <div class="equation">
        &lambda;<sub>effective</sub> = max(&lambda;, &minus;&mu;<sub>min</sub> + margin)
    </div>
    <p>with margin = 0.1 (safety buffer). This guarantees (H + &lambda;I) is positive definite
    regardless of Fisher values.</p>

    <h3>Catch-Up Iterations</h3>
    <p>After the initial Taylor update for task <em>t</em>, we perform <code>catch_up_iterations=10</code>
    refinement steps:</p>
    <ol>
        <li>Recompute the gradient <strong>g</strong> at the <em>new</em> global position
            (the cumulative Fisher stays the same)</li>
        <li>Apply the Taylor update again with the new gradient</li>
    </ol>
    <p>This is a form of fixed-point iteration: the initial update uses the gradient at the
    old position; catch-up iterations &ldquo;catch up&rdquo; to the gradient at the new position.
    Converges quickly because the Fisher (curvature) changes slowly.</p>

<pre><code>for task_t in tasks:
    w_global = taylor_update(w_global, w_expert_t, g, F_cum)
    for k in range(catch_up_iters):
        g_new = compute_gradient(w_global)     # gradient at new position
        w_global = taylor_update(w_global, w_expert_t, g_new, F_cum)
    F_cum += F_task_t
    g_cum += g_new</code></pre>

    <div class="callout success">
        <strong>Effect:</strong> Without catch-up, we get a single Newton-like step that may overshoot.
        With 10 catch-up iterations, we achieve significantly better convergence in practice,
        as the gradient is re-evaluated at the updated position.
    </div>
</section>

<!-- ═══════════════════════════════════════════════════════════
     13. EVALUATION PIPELINE
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-eval">
    <h2>13. Evaluation and Comparison Pipeline</h2>

    <h3>Evaluation Protocol</h3>
    <p>Each model (expert or consolidated) is evaluated on <strong>all</strong> tasks:</p>
    <ol>
        <li><strong>Create evaluation environment:</strong> No episodic life (full game until all lives lost),
            no reward clipping (raw game rewards), same frame stacking and preprocessing otherwise.</li>
        <li><strong>Run episodes:</strong> <code>num_episodes</code> (30 default, 2 debug) deterministic episodes.
            Action selection: argmax Q<sub>masked</sub>(s, a) with no epsilon exploration.
            Safety limit: 27,000 steps per episode (~108K frames with skip=4).</li>
        <li><strong>Report:</strong> mean, std, min, max, median reward.</li>
    </ol>

    <h3>Comparison Matrix</h3>
    <table>
        <tr><th>Method</th><th>Pong</th><th>Breakout</th><th>SpaceInvaders</th><th>Average</th></tr>
        <tr><td>Expert</td><td>E<sub>pong</sub></td><td>E<sub>break</sub></td><td>E<sub>space</sub></td><td>avg(E)</td></tr>
        <tr><td>Distillation</td><td>dist<sub>p</sub></td><td>dist<sub>b</sub></td><td>dist<sub>s</sub></td><td>avg(dist)</td></tr>
        <tr><td>HTCL</td><td>htcl<sub>p</sub></td><td>htcl<sub>b</sub></td><td>htcl<sub>s</sub></td><td>avg(htcl)</td></tr>
    </table>

    <div class="callout info">
        <strong>Note:</strong> Each Expert is evaluated only on its own task (specialist).
        Consolidated models are evaluated on all tasks.
    </div>

    <h3>Visualizations</h3>
    <ol>
        <li>Grouped bar chart &mdash; methods &times; games with error bars</li>
        <li>Performance heatmap &mdash; raw reward + percentage of expert performance</li>
        <li>Radar chart &mdash; multi-dimensional view of method strengths</li>
        <li>Box plots / violin plots &mdash; reward distribution per method/task</li>
        <li>Retention chart &mdash; horizontal bars showing % of expert performance</li>
        <li>Forgetting analysis &mdash; diverging bars showing reward degradation</li>
        <li>Summary dashboard &mdash; multi-panel overview</li>
        <li>Dot comparison &mdash; lollipop chart of expert vs. consolidated</li>
    </ol>
</section>

<!-- ═══════════════════════════════════════════════════════════
     14. HYPERPARAMETER TABLE
     ═══════════════════════════════════════════════════════════ -->
<section id="sec-hyperparams">
    <h2>14. Summary of Hyperparameters</h2>

    <table>
        <tr><th>Category</th><th>Parameter</th><th>Default</th><th>Debug</th></tr>
        <tr><td class="category cat-model" rowspan="4">Model</td>
            <td>Conv channels</td><td>[32, 64, 64]</td><td>same</td></tr>
        <tr><td>FC hidden</td><td>512</td><td>same</td></tr>
        <tr><td>Union actions</td><td>6 (computed at runtime)</td><td>same</td></tr>
        <tr><td>Total params</td><td>~1.7M</td><td>same</td></tr>

        <tr><td class="category cat-training" rowspan="8">Training</td>
            <td>Total timesteps</td><td>1,000,000</td><td>5,000</td></tr>
        <tr><td>Learning rate</td><td>1e-4</td><td>same</td></tr>
        <tr><td>Batch size</td><td>32</td><td>same</td></tr>
        <tr><td>Gamma (discount)</td><td>0.99</td><td>same</td></tr>
        <tr><td>Buffer size</td><td>200,000</td><td>1,000</td></tr>
        <tr><td>Eps decay steps</td><td>250,000</td><td>same</td></tr>
        <tr><td>Target update freq</td><td>2,500</td><td>same</td></tr>
        <tr><td>Gradient clip</td><td>10.0</td><td>same</td></tr>

        <tr><td class="category cat-norm" rowspan="2">Normalization</td>
            <td>Method</td><td>popart</td><td>same</td></tr>
        <tr><td>Momentum</td><td>0.01</td><td>same</td></tr>

        <tr><td class="category cat-distill" rowspan="5">Distillation</td>
            <td>Temperature</td><td>2.0</td><td>same</td></tr>
        <tr><td>Alpha</td><td>0.5</td><td>same</td></tr>
        <tr><td>Epochs</td><td>50</td><td>5</td></tr>
        <tr><td>Learning rate</td><td>5e-5</td><td>same</td></tr>
        <tr><td>Buffer per task</td><td>10,000</td><td>500</td></tr>

        <tr><td class="category cat-htcl" rowspan="5">HTCL</td>
            <td>Lambda</td><td>1.0</td><td>same</td></tr>
        <tr><td>Lambda auto-adjust</td><td>true</td><td>same</td></tr>
        <tr><td>Fisher samples</td><td>2,000</td><td>100</td></tr>
        <tr><td>Catch-up iters</td><td>10</td><td>same</td></tr>
        <tr><td>Eta (step size)</td><td>0.9</td><td>same</td></tr>

        <tr><td class="category cat-eval">Evaluation</td>
            <td>Episodes</td><td>30</td><td>2</td></tr>
    </table>
</section>

<!-- ═══════════════════════════════════════════════════════════
     FOOTER
     ═══════════════════════════════════════════════════════════ -->
<footer>
    CRL-Atari Technical Report &middot; Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
    &middot; <a href="https://github.com/ProtikNag/CRL-Atari" style="color: {NAVY};">GitHub Repository</a>
</footer>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report generated: {output_path}")
    print(f"  Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CRL-Atari HTML Technical Report")
    parser.add_argument("--output", type=str,
                        default="docs/CRL_Atari_Technical_Report.html",
                        help="Output HTML file path")
    args = parser.parse_args()
    generate_report(args.output)
