import asyncio
import pandas as pd
import websockets

def load_data(file_path):
    return pd.read_parquet(file_path)

async def send_data(websocket, path):
    data = load_data('./2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet')
    total_duration = 25  # Total duration of the measurement in seconds
    total_points = len(data)
    batch_size = total_points // (total_duration * 100)  # Further reduce batch size

    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data.iloc[start:end]
        try:
            await websocket.send(batch.to_json(orient='split'))
            print(f'Sent batch: {start} to {end}, size: {end-start}')
        except websockets.ConnectionClosedError as e:
            print(f'Connection closed error: {e}')
            break
        await asyncio.sleep(0.05)  # Simulate real-time delay with more frequent updates

async def main():
    async with websockets.serve(send_data, "localhost", 8765):
        print('Simulation server is waiting for dashboard connection')
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
