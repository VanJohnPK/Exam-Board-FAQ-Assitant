from agents.research_assistant import research_assistant

graph = research_assistant.get_graph(xray=True)
png_data = graph.draw_mermaid_png()
with open('state_graph.png', 'wb') as f:
    f.write(png_data)