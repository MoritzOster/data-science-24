import asyncio
import pandas as pd
import websockets
import json

def load_data(file_path):
    return pd.read_parquet(file_path)

async def send_data(websocket, path):
    data = load_data('./2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet')
    batch_size = 2000 * 10  
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data.iloc[start:end]
        await websocket.send(batch.to_json())
        await asyncio.sleep(1) 

async def main():
    async with websockets.serve(send_data, "localhost", 8765):
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())
