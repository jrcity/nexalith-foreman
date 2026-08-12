#!/bin/bash

echo "====================================================="
echo "   NEXALITH FOREMAN - OFFLINE EDGE & THERMAL TEST    "
echo "====================================================="

# Prompt for sudo early so it doesn't interrupt the test
echo "We need sudo access to change the CPU governor."
sudo -v

echo -e "\n---> SWITCHING CPU GOVERNOR TO: PERFORMANCE"
sudo cpupower frequency-set -g performance

echo -e "\n---> RUNNING EDGE TESTS (PERFORMANCE MODE)"
python3 /home/redemption/codebase/ai/nexalith-foreman/edge_test.py

echo -e "\n---> LETTING THERMALS STABILIZE (10 SECONDS)..."
sleep 10

echo -e "\n---> SWITCHING CPU GOVERNOR TO: POWERSAVE (BALANCED)"
sudo cpupower frequency-set -g powersave

echo -e "\n---> RUNNING EDGE TESTS (BALANCED MODE)"
python3 /home/redemption/codebase/ai/nexalith-foreman/edge_test.py

echo -e "\n====================================================="
echo "   TESTING COMPLETE"
echo "====================================================="
