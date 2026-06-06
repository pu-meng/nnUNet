hostname -I
ip addr | grep inet

ssh-keygen -t ed25519

cat ~/.ssh/id_ed25519.pub


echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFWZltQTpNt28P/D+Ewy6zvX8IjQCS60/ohQXRY9JR62" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

du -h --max-depth=1 | sort -hr | head -n 5

free -h
sudo dmidecode -t memory


tar -cvzf code.tar.gz \
--exclude='*/__pycache__' \
--exclude='*.sh' \
--exclude='*.ipynb' \
--exclude='*.pth' \
--exclude='*.ckpt' \
--exclude='*.pt' \
--exclude='*/.git' \
medseg_project


tar -xf "/home/PuMengYu/Task03_Liver (1).tar" -C /home/PuMengYu/