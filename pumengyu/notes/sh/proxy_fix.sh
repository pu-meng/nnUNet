#!/bin/bash
curl -s -X PUT "http://127.0.0.1:9091/proxies/%F0%9F%9A%80%20%E8%8A%82%E7%82%B9%E9%80%89%E6%8B%A9" -H "Content-Type: application/json" -d '{"name":"♻️ 自动选择"}'
echo ""
curl -x http://127.0.0.1:7891 --max-time 10 https://www.google.com -o /dev/null -w "状态码: %{http_code}\n"
