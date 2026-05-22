for node in "🇺🇸 美国-专线 · 01" "🇺🇸 美国-专线 · 02" "🇺🇸 美国 ·原生 03" "🇺🇸 美国 ·原生 04" "🇺🇸 美国 · 05 · Chatgpt/TikTok" "🇺🇸 美国 · 06 · Chatgpt/TikTok"; do
  curl -s -X PUT "http://127.0.0.1:9090/proxies/%E5%A4%A7%E5%93%A5%E4%BA%91" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$node\"}"
  code=$(curl -s https://api.anthropic.com -o /dev/null -w "%{http_code}" --max-time 8)
  echo "$node -> $code"
  [ "$code" = "404" ] && echo "✅ 通了！" && break
done