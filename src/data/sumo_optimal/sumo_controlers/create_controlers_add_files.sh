max_cycle_time=60
max_cycle_time=$(($max_cycle_time - 2))
min_cycle_time=$(($max_cycle_time - 4))
# coord: remember to cofigurate the "tls_timed.add.xml" cycle time
python "C:/Program Files (x86)/Eclipse/Sumo/tools/tlsCoordinator.py" \
    -n "optimal.net.xml" \
    -r "optimal.passenger.rou.xml" \
    -a "tls_timed.add.xml" \
    -o "tlsOffsets.add.xml"

# webster:
python "C:/Program Files (x86)/Eclipse/Sumo/tools/tlsCycleAdaptation.py" \
    -n "optimal.net.xml" \
    -r "optimal.passenger.rou.xml" \
    -y 3 \
    -a 2 \
    -g 5 \
    -C $max_cycle_time \
    -c $min_cycle_time \
    --unified-cycle \
    -o "tls_webster.add.xml" \