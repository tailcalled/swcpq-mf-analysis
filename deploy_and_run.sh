set -e
source ./secrets.sh

rsync -avz run.sh analysis.py requirements.txt sex_override.json SWCPQ-Features-Aggregated-Dataset-January2025 $DEV_PC:/home/root/swcpq-mf-analysis
ssh $DEV_PC chmod +x /home/root/swcpq-mf-analysis/run.sh
ssh $DEV_PC /home/root/swcpq-mf-analysis/run.sh | tee output.txt

rsync -avz $DEV_PC:/home/root/swcpq-mf-analysis/plots .