#!/bin/sh

# Linux data-gathering commands; adjust as necessary for your platform.
#
# Be sure to remove any information from the output that would violate
# SC's double-blind review policies.

# Anonymize user-specific information
env | sed "s/$USER/USER/g"

# Enable debug mode to show each command before it is executed
set -x

# Collect system information
lsb_release -a 2>&1 | tee -a log
uname -a 2>&1 | tee -a log
lscpu 2>&1 || cat /proc/cpuinfo 2>&1 | tee -a log
cat /proc/meminfo 2>&1 | tee -a log
inxi -F -c0 2>&1 | tee -a log
lsblk -a 2>&1 | tee -a log
lsscsi -s 2>&1 | tee -a log
module list 2>&1 | tee -a log
nvidia-smi 2>&1 | tee -a log

# Append hardware details to the log
(lshw -short -quiet -sanitize || lspci) | cat >> log

# Final message indicating the completion of data gathering
echo "Data gathering complete. Check the 'log' file for details."
