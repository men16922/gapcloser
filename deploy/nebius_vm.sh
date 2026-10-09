#!/usr/bin/env bash
# Deploy Tether to a Nebius AI Cloud CPU VM (eu-west1 by default), retrying VM creation until the account allows it
# (billing details saved). Then: copy the committed code, build the Docker image on the VM, run it on port 80.
# Needs: nebius CLI profile (nebius profile create), ~/.ssh/gapcloser_nebius, NEBIUS_API_KEY in .env.
# Usage: deploy/nebius_vm.sh [max_wait_minutes]      Stop billing afterwards: nebius compute instance delete --id <id>
set -uo pipefail
cd "$(dirname "$0")/.."
NB=~/.nebius/bin/nebius
P=${NEBIUS_PROJECT:-project-e01r9mqhpa00k484wcc2ct}          # default-project-eu-west1 (eu-north1 has 0 non-GPU vCPU quota)
SUBNET=${NEBIUS_SUBNET:-vpcsubnet-e01ph41k4q7qewq9r4}
NAME=gapcloser-demo
KEY=~/.ssh/gapcloser_nebius
WAIT_MIN=${1:-180}
log() { echo "[$(date +%H:%M:%S)] $*"; }

PUB=$(cat "$KEY.pub")
UD="#cloud-config
users:
  - name: gapcloser
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $PUB
package_update: true
packages: [docker.io]
runcmd:
  - systemctl enable --now docker
  - usermod -aG docker gapcloser
  - touch /var/tmp/cloud-init-done"

ID=$($NB compute instance get-by-name --parent-id $P --name $NAME --format json 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['metadata']['id'])" 2>/dev/null)
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
while [ -z "$ID" ]; do
  out=$($NB compute instance create --parent-id $P --name $NAME \
    --resources-platform ${NEBIUS_PLATFORM:-cpu-d3} --resources-preset 2vcpu-8gb \
    --boot-disk-attach-mode read_write --boot-disk-device-id boot \
    --boot-disk-managed-disk-name gapcloser-boot --boot-disk-managed-disk-size-gibibytes 40 \
    --boot-disk-managed-disk-type network_ssd \
    --boot-disk-managed-disk-source-image-family-image-family ubuntu22.04-driverless \
    --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET\",\"ip_address\":{},\"public_ip_address\":{}}]" \
    --cloud-init-user-data "$UD" --format json 2>&1)
  ID=$($NB compute instance get-by-name --parent-id $P --name $NAME --format json 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['metadata']['id'])" 2>/dev/null)  # create output can mix warnings into the JSON
  if [ -n "$ID" ]; then log "VM created: $ID"; break; fi
  reason=$(echo "$out" | grep -m1 -E "desc =" | cut -c1-200)
  log "VM create refused ($reason); waiting for billing to be activated"
  [ "$(date +%s)" -gt "$deadline" ] && { log "GAVE UP after $WAIT_MIN min"; exit 2; }
  sleep 60
done

IP=""
for i in $(seq 1 60); do
  IP=$($NB compute instance get --id "$ID" --format json 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
for n in d.get('status',{}).get('network_interfaces',[]):
  a=(n.get('public_ip_address') or {}).get('address','')
  if a: print(a.split('/')[0])" 2>/dev/null)
  [ -n "$IP" ] && break; sleep 10
done
[ -z "$IP" ] && { log "FAILED: no public IP"; exit 3; }
log "public IP $IP"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR gapcloser@$IP"
for i in $(seq 1 60); do $SSH "test -f /var/tmp/cloud-init-done" 2>/dev/null && break; sleep 10; done
$SSH "test -f /var/tmp/cloud-init-done" || { log "FAILED: cloud-init not finished"; exit 4; }
log "VM ready; copying code"
git archive --format=tar HEAD | $SSH "rm -rf ~/gapcloser && mkdir ~/gapcloser && tar -x -C ~/gapcloser"
grep '^NEBIUS_API_KEY=' .env | $SSH "umask 077; cat > ~/gapcloser.env"
log "building image on the VM (several minutes)"
$SSH "cd ~/gapcloser && sudo docker build -q -t gapcloser . " || { log "FAILED: docker build"; exit 5; }
$SSH "sudo docker rm -f gapcloser >/dev/null 2>&1; sudo docker run -d --name gapcloser --restart unless-stopped -p 80:7860 \
  --env-file ~/gapcloser.env -e GAPCLOSER_LLM=tokenfactory gapcloser" >/dev/null || { log "FAILED: docker run"; exit 6; }
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://$IP/studio")
  [ "$code" = "200" ] && break; sleep 5
done
log "DEPLOYED http://$IP/studio (console http://$IP/) status $code; instance $ID"
