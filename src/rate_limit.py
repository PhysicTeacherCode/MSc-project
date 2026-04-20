import asyncio
import aiohttp
import time

async def _test_batch(batch_size: int) -> bool:
    """Dispara um lote concorrente de requisições. 
    Retorna True se bateu no HTTP 429 (Too Many Requests)."""
    
    url = "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile?actor=bsky.app"
    connector = aiohttp.TCPConnector(limit=batch_size)
    
    async def fetch(session):
        async with session.get(url) as response:
            if response.status == 429:
                return True
            return False

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session) for _ in range(batch_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return any(r is True for r in results if not isinstance(r, Exception))

async def calibrate_rate_limit(max_test_concurrency=100) -> int:
    """Aumenta gradativamente a concorrência para achar o limite de requisições por segundo.
    Retorna um limite seguro para o Semáforo do asyncio."""
    print("[Rate Limit] Iniciando teste de calibração de requisições...")
    
    # Vamos testar concorrências de 10, 20, 30...
    step = 10
    current_concurrency = step
    
    safe_limit = step

    while current_concurrency <= max_test_concurrency:
        start = time.time()
        print(f"[Rate Limit] Testando lote com concorrência = {current_concurrency}...", end="\r")
        
        hit_limit = await _test_batch(current_concurrency)
        elapsed = time.time() - start
        
        if hit_limit:
            print(f"\n[Rate Limit] HTTP 429 bloqueado ao atingir {current_concurrency} requisições concorrentes.")
            # Definimos o limite seguro como a camada anterior que funcionou ou 70% do limite atingido
            safe_limit = max(step, int(current_concurrency * 0.7))
            break
        else:
            safe_limit = current_concurrency
        
        # Esperamos um pouco antes do próximo lote pra não poluir
        await asyncio.sleep(min(1.0, elapsed))
        current_concurrency += step

    if current_concurrency > max_test_concurrency:
        print(f"[Rate Limit] Não bateu em 429 nos testes. Adotando safe limit padrão = {safe_limit}.")
    
    print(f"[Rate Limit] ---> CONCORRÊNCIA SEGURA ESTABELECIDA: {safe_limit}")
    return safe_limit
