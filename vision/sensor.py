import requests

SENSOR_IP = '10.187.226.135:8080'

def get_lux(timeout=0.6):
    """
    Polls the BH1750 Ambient Light Sensor over HTTP REST.
    Returns:
        float: Lux value if valid (> 0)
        None: If sensor is offline or returning an error code (e.g. -2.0)
    """
    try:
        url = f'http://{SENSOR_IP}/lux'
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            val = float(data.get('lux', -1.0))
            if val > 0:
                return round(val, 1)
            # Negative reading indicates I2C wiring issue on DevKit
            return None
    except Exception:
        pass
    return None

if __name__ == '__main__':
    reading = get_lux()
    if reading is not None:
        print(f'BH1750 Sensor Reading: {reading} lx')
    else:
        print('BH1750 Sensor Offline / I2C error (Check SDA=21, SCL=22).')
