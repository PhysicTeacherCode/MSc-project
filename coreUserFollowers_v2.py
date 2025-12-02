# Esse programa foi criado para fazer o scrapping no API do Bluesky, coletando o nome de todos os
# followers de cada 'core_user' previamente selecionados arbitrariamente, para posterior coleta dos
# posts de cada um dos followers

from concurrent.futures import ThreadPoolExecutor, as_completed # Para evitar ficar dias rodando o código
import pandas as pd # criar tabelas
import requests # Usar o API do Bluesky

# Lista com os usuários principais (escolhidos arbitrariamente)
core_user = [
"randall.gobirds.online",
"msevelyn.bsky.social",
"lexicodex.bsky.social",
"garyrbs.bsky.social",
"jelle8591.bsky.social",
"aussieopinion.bsky.social",
"freebirdthirteen.bsky.social",
"janvan.bsky.social",
"lizettekodama.bsky.social",
"ludditebro.bsky.social",
"stellasjr.bsky.social",
"soupster16.bsky.social",
"niksea.bsky.social",
"bunnygabby.bsky.social",
"rst0868.bsky.social",
"yourcomicmuse.bsky.social",
"willcuthere.bsky.social",
"cionaod.bsky.social",
"aut-of-order.bsky.social",
"leandrot.bsky.social"
]

# -----------------------
# 1. Buscar seguidores
# -----------------------
def get_followers(actor_handle):
    url = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollowers" # URL do API do Bluesky
    cursor = None # Necesário para armazenar uma quantidade maior de usuário (limite por chamada = 100)
    followers = [] # Lista onde será armazenada os usuários

    # Chama o API e coleta os usuários (followers)
    while True:
        params = {
            "actor": actor_handle,
            "limit": 100
        }
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print("Erro:", response.status_code, response.text)
            break

        data = response.json()

        # lista de seguidores retornados na página atual
        for item in data.get("followers", []):
            followers.append(item.get("handle"))

        # pega o próximo cursor para continuar
        cursor = data.get("cursor")

        if not cursor:
            break

    return actor_handle, followers

# -----------------------
# 2. Executando
# -----------------------
followers_list = []
collumnsName = []

# Roda o programa para coletar o nome de 20 Followers ao mesmo tempo
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(get_followers, user) for user in core_user]
    for future in as_completed(futures):
        actor, users = future.result()
        followers_list.append(users)
        collumnsName.append(actor)

# Para criar o arquivo EXCEL
#df = pd.DataFrame(followers_list).T
#df.columns = collumnsName
#df.to_excel("all_followers.xlsx", index=False)

# -----------------------
# 3. Visualizando
# -----------------------
for x in followers_list:

   print(len(x),x)
