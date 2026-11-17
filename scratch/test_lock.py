import asyncio

async def test():
    print("Testing lock in current loop...", flush=True)
    lock = asyncio.Lock()
    async with lock:
        print("Acquired lock successfully!", flush=True)

asyncio.run(test())

