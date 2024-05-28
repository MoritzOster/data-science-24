import streamlit as st
import pandas as pd
import asyncio
import websockets
import json
import time

async def fetch_data():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            yield pd.read_json(data)

st.title('Real-Time AE Signal Visualization')
st.write("Simulating real-time data from AE measurements.")

chart = st.line_chart()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def display_data():
    async for batch in fetch_data():
        chart.add_rows(batch)

# display_data coroutine
loop.run_until_complete(display_data())
