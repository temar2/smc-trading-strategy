#!/usr/bin/env python3
"""
Test rapide de l'API TwelveData
"""

import requests

def test_api_key(api_key):
    """Test simple de l'API key"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "GBP/USD",
        "interval": "1day",
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
        "outputsize": "5"  # Juste 5 points pour tester
    }
    
    try:
        print(f"🔍 Test API key: {api_key[:8]}...")
        response = requests.get(url, params=params, timeout=10)
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "values" in data and data["values"]:
                print(f"✅ API OK - {len(data['values'])} points reçus")
                return True
            else:
                print(f"❌ API Error: {data}")
                return False
        else:
            print(f"❌ HTTP Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    # Test avec l'API key du script
    api_key = "8af42105d7754290bc090dfb3a6ca6d4"
    test_api_key(api_key)
