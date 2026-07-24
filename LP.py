import pulp
import math
import time
import serial
import requests
import numpy as np

SECS_PER_DAY = 300.0
TICKS_PER_DAY = 60
Tick_duration = 5
BASE_DEMAND_SCALING = 0.02
BASE_DEMAND_PROFILE = [(0, 25), (10, 25), (20, 100), (50, 100), (TICKS_PER_DAY, 25)]
PRICE_MIN = 10
BASE_PRICE = 10.0
BUY_RATIO = 0.5
PRICE_SOLAR_DEP = 1.0
P_maxLoad = 3

C_CAP      = 0.5
V_FLOOR    = 10.0
V_CEILing  = 15.5
ESR        = 2.0
I_MAX      = 0.6
T_TICK     = 5.0
Emin = 0.5 * C_CAP * V_FLOOR**2
Emax = 0.5 * C_CAP * V_CEILing**2

# flat efficiency measured from hardware
efficiency_store   = 0.765
efficiency_extract = 0.769

# empirical caps from hardware testing
max_store   = 20.0
max_extract = 15.0

error_margin  = 0
maintain_loss = 0
overshoot     = 1.5

API_URL    = 'https://icelec50015.azurewebsites.net'
SERIAL_PORT = 'COM3'
BAUD_RATE   = 115200

STATIC_LUT = {
    0: {'vmmp': 0.0,   'immp': 0.0},
    10: {'vmmp': 2.352, 'immp': 0.064},
    20: {'vmmp': 4.704, 'immp': 0.128},
    30: {'vmmp': 5.96,  'immp': 0.184},
    40: {'vmmp': 6.12,  'immp': 0.232},
    49: {'vmmp': 6.264, 'immp': 0.275},
    58: {'vmmp': 6.306, 'immp': 0.318},
    66: {'vmmp': 6.331, 'immp': 0.357},
    74: {'vmmp': 6.357, 'immp': 0.395},
    80: {'vmmp': 6.392, 'immp': 0.41},
    86: {'vmmp': 6.43,  'immp': 0.422},
    91: {'vmmp': 6.462, 'immp': 0.432},
    95: {'vmmp': 6.488, 'immp': 0.44},
    97: {'vmmp': 6.501, 'immp': 0.444},
    99: {'vmmp': 6.514, 'immp': 0.448},
    100: {'vmmp': 6.52, 'immp': 0.45}
}


def energy(fallback=Emin):
    tripped = False
    try:
        pico.reset_input_buffer() # controller spits out values constantly, only want the commanded one so remove old ones
        pico.write(b"P\n")
        time.sleep(0.05)          # give the Pico time to respond before we read
        line = pico.readlines()

        for i in reversed(line):  # since it is oldest to newest want newest first
            i = i.decode("utf-8").strip()
            if "E=" in i and "mode=" in i:
                tripped = "trip=True" in i
                energy_str = i.split("E=")[1].split("J")[0] # the way of getting the value
                return float(energy_str), tripped
        print(f"could not parse energy response, defaulting to {fallback}J")
        return fallback, tripped

    except (ValueError, OSError, IndexError) as e:
        print(f"hardware read failed ({e}), using fallback {fallback}J")
        return fallback, False


def send_command(cmd):
    try:
        pico.write(cmd)
    except OSError as e:
        print(f"serial write failed: {e}")


def api_price():
    try:
        return requests.get(f'{API_URL}/price', timeout=3).json()
    except Exception as e:
        print(f"API price failed: {e}")
        return None


def api_deferables():
    try:
        return requests.get(f'{API_URL}/deferables', timeout=3).json()
    except Exception as e:
        print(f"API deferables failed: {e}")
        return None


def getSunlight(tick):
    if tick < 15 or tick >= 45:
        return 0
    return int(math.sin((tick - 15) * math.pi / 30) * 100)


def raw_pv(live_sun):
    nearest_key = min(STATIC_LUT.keys(), key=lambda k: abs(k - live_sun))
    params = STATIC_LUT[nearest_key]
    return params['vmmp'] * params['immp']


def solar_power(tick):
    return 0.85 * raw_pv(getSunlight(tick))


def getBaseDemand(tick):
    lastp = (0, 0)
    for p in BASE_DEMAND_PROFILE:
        if tick < p[0]:
            return int(float(tick - lastp[0]) / (float(p[0] - lastp[0])) * (p[1] - lastp[1]) + lastp[1])
        else:
            lastp = p
    return lastp[1]


def get_expected_prices():
    # simple forecast: base price shifted by demand/solar profile, floored at PRICE_MIN
    return [max(PRICE_MIN, BASE_PRICE + (getBaseDemand(t) - getSunlight(t)) * PRICE_SOLAR_DEP)
            for t in range(TICKS_PER_DAY)]


def solver(solar, demand, current_charge, cost, sell_price, E_max, energy_remaining, deferables, current_tick, remaining_ticks=60):
    h = min(remaining_ticks, 60)
    current_charge = np.clip(current_charge, Emin + error_margin, Emax - error_margin)

    # flat caps — same for all ticks, no voltage dependence
    max_store_cap   = [max_store   for _ in range(h)]
    max_extract_cap = [max_extract for _ in range(h)]

    score = pulp.LpProblem("Battery_Optimiser", pulp.LpMinimize)

    L = {}
    for j in range(len(deferables)):
        L[j] = pulp.LpVariable.dicts(f'L{j}', range(h), lowBound=0)

    Pb = pulp.LpVariable.dicts("Pb", range(h), lowBound=0) # power bought from and sold to the grid
    Ps = pulp.LpVariable.dicts("Ps", range(h), lowBound=0)

    Ec = [pulp.LpVariable(f"Ec_{i}", lowBound=0, upBound=max_store_cap[i])   for i in range(h)] # how much to charge and discharge limited by the empirical cap
    Ed = [pulp.LpVariable(f"Ed_{i}", lowBound=0, upBound=max_extract_cap[i]) for i in range(h)]

    Eb = [pulp.LpVariable(f"Eb_{i}", lowBound=Emin + error_margin, upBound=Emax - error_margin)
          for i in range(h + 1)] # h+1 because for h ticks there are h+1 energy points
    DISCHARGEmargin = 0.3

    score += pulp.lpSum(Pb[i] * cost[i] - Ps[i] * sell_price[i] + DISCHARGEmargin * Ed[i]
                        for i in range(h)) # objective: minimise net cost

    score += (Eb[0] == current_charge) # 0 is the current tick so make that equal to the TRUE value of energy in the cap
    score += (Eb[h] == Emin + error_margin) # we want the cap empty at the end of a cycle

    for j in range(len(deferables)):
        score += pulp.lpSum(L[j][i] * Tick_duration for i in range(h)) == energy_remaining[j]

    for i in range(h):
        tick_load = pulp.lpSum(L[j][i] for j in range(len(deferables)))
        headroom  = max(0, P_maxLoad - demand[i])
        score += (tick_load <= headroom)
        for j in range(len(deferables)):
            score += (L[j][i] <= P_maxLoad)
            job_dict = deferables[j]
            start = job_dict['start']
            end   = job_dict['end']
            if i + current_tick > end or i + current_tick < start:
                score += (L[j][i] == 0)

        score += (Eb[i + 1] == Eb[i] + Ec[i] - Ed[i] + maintain_loss)

        # grid power balance with flat efficiency
        score += (Pb[i] + solar[i] + (Ed[i] / Tick_duration) * efficiency_extract ==
                  Ps[i] + demand[i] + tick_load + (Ec[i] / Tick_duration) * (1.0 / efficiency_store))

    score.solve(pulp.HiGHS(msg=0))

    current_tickLoad = {j: max(0.0, pulp.value(L[j][0])) for j in range(len(deferables))}
    net_grid_power = (pulp.value(Pb[0])) - (pulp.value(Ps[0]))
    store_J   = max(0.0, pulp.value(Ec[0]))
    extract_J = max(0.0, pulp.value(Ed[0]))
    new_E     = float(np.clip(current_charge + store_J - extract_J, Emin, Emax))

    return net_grid_power, new_E, store_J, extract_J, current_tickLoad


if __name__ == '__main__':

    pico = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print(f"serial port {SERIAL_PORT} open at {BAUD_RATE} baud")
    send_command(b"H\n")  # ensure known state
    time.sleep(0.1)
    send_command(b"U\n")
    time.sleep(0.2)

    current_E        = Emin
    cost             = 0
    sell_price_total = 0
    last_tick        = -1
    last_day         = -1
    deferables       = []
    energy_remaining = {}

    try:
        while True:
            print("polling...")
            data = api_price()
            if data is None:
                time.sleep(1)
                continue

            current_day  = data['day']
            current_tick = data['tick']
            price        = data['sell_price']
            buy_now      = data['buy_price']
            raw_demand   = data.get('demand', getBaseDemand(current_tick) * BASE_DEMAND_SCALING) # safety fallback to forecast if not present

            if current_day != last_day:
                deferables = api_deferables() # reget the deferables each new day
                if deferables is None:
                    time.sleep(1)
                    continue
                last_day              = current_day # ensure we dont reget the deferables on every poll
                expected_prices       = get_expected_prices()
                energy_remaining      = {i: d['energy'] for i, d in enumerate(deferables)}
                print(f"new day {current_day} | jobs: {energy_remaining}")

            if last_tick != current_tick:
                demand  = raw_demand
                Psolar  = solar_power(current_tick)
                current_E, tripped = energy(fallback=current_E) # gets energy from cap controller
                if tripped:
                    print("hardware tripped, resetting")
                    send_command(b"H\n")
                    time.sleep(0.1)
                    send_command(b"U\n")

                remaining = TICKS_PER_DAY - current_tick
                solar_arr, demand_arr, cost_arr, sell_arr = [], [], [], []
                for i in range(remaining):
                    t = current_tick + i
                    if i == 0:
                        solar_arr.append(Psolar); demand_arr.append(demand)
                        cost_arr.append(price);   sell_arr.append(buy_now)
                    else:
                        solar_arr.append(solar_power(t))
                        demand_arr.append(getBaseDemand(t) * BASE_DEMAND_SCALING)
                        cost_arr.append(expected_prices[t]  if t < len(expected_prices) else expected_prices[-1])
                        sell_arr.append((expected_prices[t] if t < len(expected_prices) else expected_prices[-1]) * BUY_RATIO)

                Pgrid, new_E, store_J, extract_J, loads_this_tick = solver(
                    solar_arr, demand_arr, current_E, cost_arr, sell_arr, Emax,
                    energy_remaining, deferables, current_tick, remaining_ticks=remaining)

                for j in range(len(deferables)):
                    energy_remaining[j] = max(0.0, energy_remaining[j] - loads_this_tick[j] * Tick_duration)

                store_cmd   = max(0.0, store_J   - overshoot)
                extract_cmd = max(0.0, extract_J - overshoot)

                if store_cmd >= 1.0:
                    send_command(f"S{store_cmd:.2f}\n".encode())
                    action = "STORE"
                elif extract_cmd >= 1.0:
                    send_command(f"E{extract_cmd:.2f}\n".encode())
                    action = "EXTRACT"
                else:
                    send_command(b"U\n")
                    action = "MAINTAIN"

                current_E = new_E

                if Pgrid > 0:
                    cost += Pgrid * Tick_duration * price
                else:
                    sell_price_total += Pgrid * Tick_duration * buy_now

                last_tick      = current_tick
                total_def_load = sum(loads_this_tick.values())
                V_now          = np.sqrt(2 * current_E / C_CAP)
                print(f"tick {current_tick:02d} | E: {current_E:.1f}J ({V_now:.2f}V) | solar: {Psolar:.3f}W | "
                      f"demand: {demand:.3f}W | price: {price:.0f} | {action} | "
                      f"def: {total_def_load:.2f}W | P&L: {(cost + sell_price_total):.2f}")

            time.sleep(1)

    except KeyboardInterrupt:
        print(f"sell: {sell_price_total:.2f}  cost: {cost:.2f}  P&L: {(cost+sell_price_total):.2f}")
        send_command(b"U\n")
        pico.close()
        print("serial port closed")
