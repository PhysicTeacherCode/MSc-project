
CREATE TABLE "freebirdthirteen.bsky.social_palavras_desvio" AS (
WITH palavras_idade AS (
    SELECT unnest(string_to_array(regexp_replace(lower("post"), '[^[:alpha:]]', ' ', 'g'), ' ')) AS palavras,
           DATE("date") - DATE('2023-01-01') AS idade
    FROM "freebirdthirteen.bsky.social_posts"
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