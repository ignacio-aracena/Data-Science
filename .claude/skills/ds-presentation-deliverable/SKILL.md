---
name: ds-presentation-deliverable
description: Use when assembling the final 20-slide PPTX deliverable for an academic DS project. Builds the narrative arc (question → method → results → insights → limitations → AI usage), honors slide-count limit, enforces no-names/no-code rules, and generates the AI usage log slide with literal prompts. Outputs slides/outline.md + slides/final.pptx + docs/ai-usage-log.md. Triggers when user mentions "armar slides", "presentación final", "PPTX entregable".
---

# DS Presentation Deliverable

## Overview

The consigna penalizes >20 slides, names, code, and superficial AI-usage descriptions. This skill enforces those constraints. The narrative arc is fixed; only the slide-by-slide content varies per project.

## When to use

- After business insights are locked (`reports/insights.md` exists)
- User says "armar slides" / "presentación final" / "PPTX"

Do NOT use:
- Before insights are done (no content to present)
- For internal documentation (different audience, different format)

## The 20-slide narrative arc (FIXED structure)

Slides marked `*` count toward the 20-slide limit. Portada/índice/refs no cuentan.

| # | Title | Type | Counts? | Content |
|---|---|---|---|---|
| - | Portada | — | NO | Título, materia, semestre. NO nombres. |
| - | Índice | — | NO | Bullet list de secciones. |
| 1* | Pregunta de negocio + actor + contexto | content | YES | Pregunta principal, actor target, restricción metodológica. |
| 2* | Dataset + integración | content | YES | Fuentes, volumen, diagrama de integración. |
| 3* | EDA-1: targets | figure+caption | YES | Distribución target, motiva arquitectura. |
| 4* | EDA-2: features ↔ targets | figure+caption | YES | Top features predictivas, justifica FE. |
| 5* | EDA-3: cross-source | figure+caption | YES | Validación entre fuentes. |
| 6* | Feature engineering trazable | table | YES | Features finales con `source_finding_id` visible. |
| 7* | Métricas elegidas | content | YES | Métricas + justificación derivada de pregunta. |
| 8* | Baselines | table | YES | B0/B1/B2 con métricas. Bench a vencer. |
| 9* | NN architecture | diagram | YES | v_final arch. |
| 10* | Iteración v1→vN | table | YES | Cada versión + cambio + métrica + análisis errores. |
| 11* | v_final detalle | figure+caption | YES | Training curves + precision-recall si aplica. |
| 12* | Final TEST eval | table | YES | Comparación val→test sin degradación. |
| 13* | Análisis errores 3 capas | content | YES | Layer 1/2/3 del último análisis. |
| 14* | Business formula | content | YES | Fórmulas + supuestos declarados. |
| 15* | Mapa / ranking (slide estrella) | figure | YES | Visualización principal. |
| 16* | Recomendaciones / portfolios | figure+table | YES | Top-K + perfiles. |
| 17* | Insights de mercado | content | YES | 3-5 takeaways. Conclusión honesta. |
| 18* | Limitaciones honestas | content | YES | 5-10 limitaciones declaradas. |
| 19* | Uso de IA generativa | content | YES | 3 momentos clave + reflexión. |
| 20* | Conclusión + próximos pasos | content | YES | Cierre. |
| - | Referencias | — | NO | Bibliografía. |

Total contables: 20 (cabe en el límite).

## Workflow

1. **Build outline** as `slides/outline.md` with the 20-slide structure populated from project artifacts (`reports/insights.md`, `reports/eda_findings.json`, `reports/v{N}_metrics.json`, etc.)
2. **Manifest the figures** in `slides/figures_manifest.json`: `{ slide_n: path_to_figure }`
3. **Generate PPTX skeleton** with `python-pptx`:
   ```python
   from pptx import Presentation
   from pptx.util import Inches, Pt
   
   prs = Presentation()
   prs.slide_width = Inches(13.33)
   prs.slide_height = Inches(7.5)  # 16:9
   
   # Slide 1: title
   layout = prs.slide_layouts[0]
   slide = prs.slides.add_slide(layout)
   slide.shapes.title.text = "¿..."
   slide.placeholders[1].text = "Ciencia de Datos · UdeSA · 1er sem 2026"
   
   # ... (idem 19 more content slides)
   
   prs.save("slides/final.pptx")
   ```
4. **Polish manual** (15-20 min en PowerPoint/Keynote): ajustar tamaños, captions, colores.
5. **AI usage log** (`docs/ai-usage-log.md`) con prompts LITERALES:
   ```markdown
   # AI Usage Log
   
   ## Tool: Claude Opus 4.7 + superpowers plugin (Anthropic)
   
   ## Moment 1: Brainstorming estructurado
   
  ### Prompt literal
  > "<pegá el prompt textual que usaste para esta decisión>"
   
   ### Response (summarized)
   > Claude proposed Approach A (skills-first) vs B (JIT) vs C (parallel). Recommended B but accepted user's choice of A.
   
   ### Utility: useful — accelerated design alignment in ~30 min vs ~2hs alone.
   
   ## Moment 2: ...
   
   ## Moment 3 (NOT useful — declared explicitly):
   
   ### Prompt
   > "...sample EDA approach..."
   
   ### Response
   > Claude suggested generic pandas-profiling output.
   
   ### Why not useful: too generic, didn't connect to FE. Iterated to `eda-with-narrative` skill.
   ```

## Output spec

- `slides/outline.md` — 20-slide structured markdown
- `slides/figures_manifest.json` — figure paths per slide
- `slides/final.pptx` — generated PPTX
- `docs/ai-usage-log.md` — prompts literales + utilidad evaluada

## <EXTREMELY-IMPORTANT> Rules

1. **≤ 20 content slides.** Penalty if exceeded.
2. **SIN nombres en NINGUNA slide.** Anonimato peer-review.
3. **SIN código en slides.** Consigna §2 explícito.
4. **Cada figura con caption interpretable.** Lector no-técnico debe entenderla.
5. **AI usage log con prompts LITERALES.** No resúmenes. Incluye al menos UN caso "respuesta NO útil".
6. **Honest conclusion.** Si los datos dicen "no invertir" — la slide 17 lo dice.

## Auto-review before handoff

Before declaring slides done (per user feedback `feedback_revalidate_outputs`):
1. Slide count contables = 20 exactly (or less); portada/índice/refs not counted
2. Grep for own name / surnames in `slides/outline.md` and `final.pptx` text — zero hits
3. No fenced code blocks (```` ``` ````) anywhere in slides
4. Every figure has a caption (not just a title)
5. AI usage log has ≥ 3 literal prompts (`>` quoted), including at least one "respuesta NO útil" case
6. Numbers in slides match numbers in `reports/*.json` (no drift between artifacts and slides)

If any check fails, fix BEFORE export. The PPTX is what gets graded.

## Red flags

| Thought | Reality |
|---|---|
| "21 slides, ya está, lo dejamos" | Penalización. Recortar. |
| "Pongo mi nombre como autor en footer" | Romper anonimato peer-review. NO. |
| "El AI log ya lo armo desde la memoria" | Prompts LITERALES. Si no los tenés, reconstruirlos del transcript. |
| "Caption corto, se lee el gráfico" | Profesor evalúa interpretabilidad del caption. Frase completa. |
| "Conclusión positiva queda mejor" | Conclusión honesta es lo evaluado. |
