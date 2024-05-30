import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import asyncio
import websockets

# Function to asynchronously fetch data from the WebSocket
async def fetch_data():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                data = await websocket.recv()
                batch = pd.read_json(data, orient='split')
                yield batch
            except websockets.ConnectionClosedError as e:
                print(f'Connection closed error: {e}')
                break

# Streamlit UI
st.title('Real-Time AE Signal Visualization')
st.write("Simulating real-time data from AE measurements.")

# Initialize Plotly figure
fig = go.FigureWidget()
scatter = fig.add_scatter(x=[], y=[])

plotly_chart = st.plotly_chart(fig)

# Function to update the plot with new data
def update_plot(fig, new_data):
    fig.data[0].x = list(fig.data[0].x) + list(new_data.index)
    fig.data[0].y = list(fig.data[0].y) + list(new_data['AEKi_rate2000000_clipping0_batch0'])

# Initialize the Streamlit async loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Fetch and display data
async def display_data():
    async for batch in fetch_data():
        update_plot(fig, batch)
        plotly_chart.plotly_chart(fig)

# Run the display_data coroutine
loop.run_until_complete(display_data())
