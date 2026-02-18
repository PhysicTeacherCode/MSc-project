-- Visão geral do database
SELECT * FROM "lexicodex.bsky.social_posts";

-- Visão geral dos usuários
SELECT COUNT(DISTINCT "user") AS n_usuários
FROM "lexicodex.bsky.social_posts";

-- Visão geral dos posts
SELECT COUNT(*) AS n_posts
FROM "lexicodex.bsky.social_posts";

-- Visão geral dos posts por usuário
SELECT "user", COUNT(*) AS n_posts
FROM "lexicodex.bsky.social_posts"
GROUP BY "user"
ORDER BY n_posts DESC;

-- Visão geral dos posts por data
SELECT DATE("date") AS dia, COUNT(*) AS n_posts
FROM "lexicodex.bsky.social_posts"
GROUP BY dia
ORDER BY n_posts DESC;

-- Visão geral dos posts por mês
SELECT EXTRACT(MONTH FROM DATE("date")) AS mês, COUNT(*) AS n_posts
FROM "lexicodex.bsky.social_posts"
GROUP BY mês
ORDER BY n_posts DESC;

-- Visão geral dos posts por ano
SELECT EXTRACT(YEAR FROM DATE("date")) AS ano, COUNT(*) AS n_posts
FROM "lexicodex.bsky.social_posts"
GROUP BY ano
ORDER BY n_posts DESC;

-- Pegando palavras dos posts
SELECT unnest(string_to_array(regexp_replace(lower("post"), '[^[:alpha:]]', ' ', 'g'), ' ')) AS palavras, count(*)
FROM "lexicodex.bsky.social_posts"
GROUP BY palavras
ORDER BY count DESC;

-- Idade de cada palavra
CREATE TABLE "lexicodex.bsky.social_palavras_desvio" AS (
WITH palavras_idade AS (
    SELECT unnest(string_to_array(regexp_replace(lower("post"), '[^[:alpha:]]', ' ', 'g'), ' ')) AS palavras,
           DATE("date") - DATE('2023-01-01') AS idade
    FROM "lexicodex.bsky.social_posts"
),
com_minimo AS (
    SELECT palavras,
           idade,
           idade - MIN(idade) OVER (PARTITION BY palavras) AS dias_entre_ocorrencias
    FROM palavras_idade
    WHERE palavras != ''
)
SELECT palavras,
       count(*) AS n_ocorrencias,
       STDDEV(dias_entre_ocorrencias) AS desvio_padrao
FROM com_minimo
GROUP BY palavras
HAVING COUNT(*) > 2
ORDER BY desvio_padrao DESC
);