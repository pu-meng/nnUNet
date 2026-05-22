
# 修改端口
sed -i 's/mixed-port: 7890/mixed-port: 7891/' ~/.config/clash/config.yaml
sed -i 's/external-controller: :9090/external-controller: 127.0.0.1:9091/' ~/.config/clash/config.yaml

# 重启 clash
kill $(pgrep -u PuMengYu clash)
sleep 2
/home/PuMengYu/clash -f ~/.config/clash/config.yaml > ~/clash.log 2>&1 &