import pandas as pd
import numpy as np
import plotly.io as pio

def generate_html_report(sections, title="Quantitative Report", filename="report.html"):
    """
    Generates a flexible, cross-browser HTML report from a list of sections.
    """
    print("Assembling HTML report...")

    sidebar_html = ""
    main_content_html = ""
    toc_links = []
    js_added = False  

    for i, section in enumerate(sections):
        section_id = f"section-{i}"
        toc_links.append(f'<li><a href="#{section_id}">{section["title"]}</a></li>')

        # --- Sidebar Content ---
        if "sidebar" in section:
            for item in section["sidebar"]:
                sidebar_html += f'<h2>{item["title"]}</h2>'
                if item["type"] == "metrics":
                    df = pd.DataFrame.from_dict(item["data"], orient='index', columns=['Value'])
                    sidebar_html += df.to_html(header=False, classes='metrics-table')
                elif item["type"] == "table_html":
                    sidebar_html += item["data"]

        # --- Main Content ---
        main_content_html += f'<div id="{section_id}" class="report-section">'
        main_content_html += f'<h1>{section["title"]}</h1>'
        if "description" in section:
            main_content_html += f'<p>{section["description"]}</p>'

        for item in section["main_content"]:
            item_class = "plot-item" if item["type"] == "plot" else "table-item"
            main_content_html += f'<div class="{item_class}">'
            if "title" in item:
                main_content_html += f'<h2>{item["title"]}</h2>'

            if item["type"] == "plot":
                if not js_added:
                    js_load_strategy = 'cdn'
                    js_added = True
                else:
                    js_load_strategy = False

                main_content_html += pio.to_html(
                    item["data"],
                    full_html=False,
                    include_plotlyjs=js_load_strategy
                )

            elif item["type"] == "table_html":
                main_content_html += item["data"]

            elif item["type"] == "metrics_grid":
                main_content_html += '<div class="metrics-grid">'
                for name, data in item["data"].items():
                    main_content_html += '<div class="metrics-card">'
                    main_content_html += f"<h2>{name} Portfolio</h2>"

                    weights_df = pd.DataFrame.from_dict(
                        data['weights_dict'], orient='index', columns=['Weight']
                    )
                    weights_df['Weight'] = weights_df['Weight'].map(lambda x: f"{x:.2%}")
                    main_content_html += "<h3>Optimal Weights</h3>"
                    main_content_html += weights_df.to_html(classes='metrics-table')

                    metrics_df = pd.DataFrame.from_dict(data['metrics'], orient='index', columns=['Value'])
                    main_content_html += "<h3>Performance Metrics</h3>"
                    main_content_html += metrics_df.to_html(header=False, classes='metrics-table')
                    main_content_html += "</div>"
                main_content_html += '</div>'

            main_content_html += '</div>'
        main_content_html += '</div>'

    # --- Build Final HTML ---
    html_template = f"""
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>{title}</title>
        <style>
            /* Normalize + base styles */
            *, *::before, *::after {{ box-sizing: border-box; }}
            html, body {{ margin: 0; padding: 0; }}
            
            :root {{
                --bg-color: #FFFFFF; --card-color: #F8F9FA; --text-color: #212529;
                --text-color-muted: #6C757D; --border-color: #DEE2E6; --accent-color: #007BFF;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background-color: var(--bg-color); color: var(--text-color); margin: 0; padding: 16px;
            }}
            .container {{
                display: flex; flex-direction: row; align-items: flex-start;
                justify-content: flex-start; max-width: 1800px; margin: 0 auto; padding: 10px; gap: 30px;
            }}
            h1 {{
                text-align: center; color: var(--accent-color); font-size: 2.5em;
                margin-bottom: 24px; width: 100%;
            }}
            .report-section {{
                margin-bottom: 30px; padding-bottom: 20px;
                border-bottom: 2px solid var(--accent-color);
            }}
            .report-section h1 {{
                text-align: left; font-size: 2em; color: var(--text-color); margin-bottom: 10px;
            }}
            h2 {{ color: var(--text-color-muted); border-bottom: 2px solid var(--border-color); padding-bottom: 5px; }}
            h3 {{ color: var(--text-color); }}
            .sidebar-container {{
                flex: 0 0 350px;
                position: sticky; top: 20px;
                align-self: flex-start;
                max-height: 90vh;
                overflow-y: auto;
            }}
            .toc {{
                background-color: var(--card-color); border: 1px solid var(--border-color);
                border-radius: 8px; padding: 15px; margin-bottom: 20px;
            }}
            .toc ul {{ list-style-type: none; padding-left: 10px; }}
            .toc li {{ margin-bottom: 10px; }}
            .toc a {{ text-decoration: none; color: var(--accent-color); font-weight: 500; }}
            .toc a:hover {{ text-decoration: underline; }}
            .metrics-table {{
                width: 100%; border-collapse: collapse; margin-bottom: 20px;
            }}
            .metrics-table th {{ background-color: #E9ECEF; padding: 10px 12px; text-align: left; }}
            .metrics-table tr:nth-child(odd) {{ background-color: var(--card-color); }}
            .metrics-table td {{ padding: 10px 12px; border: 1px solid var(--border-color); }}
            .metrics-table td:first-child {{ font-weight: 600; color: var(--text-color-muted); }}
            .metrics-table td:not(:first-child) {{ text-align: right; font-weight: 700; }}
            .plots-container {{
                flex: 1; display: flex; flex-direction: column; gap: 20px;
            }}
            .plot-item, .table-item {{
                background-color: var(--card-color); border: 1px solid var(--border-color);
                border-radius: 8px; padding: 20px;
            }}
            .metrics-grid {{
                display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;
            }}
            .metrics-card {{
                background-color: var(--bg-color); border: 1px solid var(--border-color);
                border-radius: 8px; padding: 20px;
            }}
            .report-section h1, .report-section h2, .report-section h3 {{
                word-wrap: break-word;
            }}
            iframe, div.plotly-graph-div {{
                width: 100% !important;
                height: auto !important;
            }}
        </style>
    </head>
    <body>
        <noscript>
            <div style="background: #ffdddd; padding: 10px; border: 1px solid red;">
                ⚠️ This report requires JavaScript to display interactive plots. Please enable it.
            </div>
        </noscript>
        <div class="container">
            <div class="sidebar-container">
                <div class="toc">
                    <h2>Navigation</h2>
                    <ul>{ "".join(toc_links) }</ul>
                </div>
                {sidebar_html}
            </div>
            <div class="plots-container">
                <h1>{title}</h1>
                {main_content_html}
            </div>
        </div>
    </body>
    </html>
    """

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"✅ Report successfully generated: {filename}")
    except Exception as e:
        print(f"❌ Error writing HTML file: {e}")