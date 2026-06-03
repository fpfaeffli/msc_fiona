# .bashrc

if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

source /etc/profile.d/modules.sh

echo "[bashrc] Loading conda..."
module load conda/2022

echo "[bashrc] Activating iacpy3_2022..."
conda activate iacpy3_2022
