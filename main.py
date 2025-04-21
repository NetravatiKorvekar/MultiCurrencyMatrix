import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
from PIL import Image
import base64
import io
import uuid

# Set page config with a more engaging title and icon
st.set_page_config(
    page_title="MultiCurrencyMatrix",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "quick"
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'conversion_history' not in st.session_state:
        st.session_state.conversion_history = []
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'favorite_currencies' not in st.session_state:
        st.session_state.favorite_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'rates' not in st.session_state:
        st.session_state.rates = {}
    if 'crypto_rates' not in st.session_state:
        st.session_state.crypto_rates = {}
    if 'default_from' not in st.session_state:
        st.session_state.default_from = 'USD'
    if 'default_to' not in st.session_state:
        st.session_state.default_to = 'EUR'

initialize_session_state()

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
def local_css():
    st.markdown("""
   <style>
/* Overall app styling */
.main {
    background-color: #f8f9fa;
    color: #212529;
    padding: 1rem;
}
.dark-mode .main {
    background-color: #212529;
    color: #f8f9fa;
}

h1 {
    color: #212529;
}
.dark-mode h1 {
    color: #f8f9fa;
}
/* Card-like containers for main content */
.content-card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    color: #212529;
}
.dark-mode .content-card {
    background-color: #343a40;
    box-shadow: 0 4px 6px rgba(0,0,0,0.4);
    color: #f8f9fa;
}

/* Improved button styling */
.stButton>button {
    background-color: #4361ee;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #3a56d4;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.dark-mode .stButton>button {
    background-color: #4895ef;
}

/* Form elements styling */
.stSelectbox>div>div, .stNumberInput>div>div {
    background-color: #ffffff;
    color: #212529;
    border-radius: 8px;
    border: 1px solid #ced4da;
}
.dark-mode .stSelectbox>div>div, .dark-mode .stNumberInput>div>div {
    background-color: #495057;
    color: #ffffff;
    border: 1px solid #6c757d;
}

/* Sidebar improvements */
.sidebar .sidebar-content {
    background-color: #e9ecef;
    padding: 15px;
    color: #212529;
}
.dark-mode .sidebar .sidebar-content {
    background-color: #343a40;
    color: #f8f9fa;
}

/* Result display highlight */
.result-display {
    font-size: 2rem;
    font-weight: bold;
    color: #4361ee;
    padding: 10px;
    background-color: rgba(67, 97, 238, 0.1);
    border-radius: 8px;
    text-align: center;
}
.dark-mode .result-display {
    color: #4895ef;
    background-color: rgba(72, 149, 239, 0.2);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    background-color: #f1f3f5;
    color: #212529;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #ffffff;
    border-bottom: none;
    font-weight: bold;
    color: #000000;
}
.dark-mode .stTabs [data-baseweb="tab"] {
    background-color: #343a40;
    color: #f8f9fa;
}
.dark-mode .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #495057;
    font-weight: bold;
}

/* For alerts and notifications */
.alert-box {
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 4px solid;
}
.alert-info {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
}
.alert-success {
    background-color: #e8f5e9;
    border-left-color: #4caf50;
}
.alert-warning {
    background-color: #fff8e1;
    border-left-color: #ff9800;
}
.dark-mode .alert-info {
    background-color: rgba(33, 150, 243, 0.2);
}
.dark-mode .alert-success {
    background-color: rgba(76, 175, 80, 0.2);
}
.dark-mode .alert-warning {
    background-color: rgba(255, 152, 0, 0.2);
}
</style>

    """, unsafe_allow_html=True)

local_css()

# Apply dark mode if enabled
if st.session_state.dark_mode:
    st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
    dark_mode_wrapper_open = True
else:
    dark_mode_wrapper_open = False

def get_exchange_rates(base_currency='USD'):
    """Fetch exchange rates from API with improved error handling"""
    try:
        # Using Exchange Rate API (free tier)
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url, timeout=10)  # Added timeout
        
        if response.status_code != 200:
            raise Exception(f"API returned status code {response.status_code}")
            
        data = response.json()
        
        # Save the data locally for offline use
        with open(f'data/exchange_rates_{base_currency}.json', 'w') as f:
            json.dump(data, f)
        
        return data['rates']
    except Exception as e:
        # Try to load from local file if available
        try:
            with open(f'data/exchange_rates_{base_currency}.json', 'r') as f:
                data = json.load(f)
            st.warning("⚠️ Using cached exchange rates (offline mode)")
            return data['rates']
        except:
            st.error(f"❌ Could not load exchange rates: {e}. Please check your internet connection.")
            return {}

def get_crypto_rates():
    """Fetch cryptocurrency rates with improved error handling"""
    try:
        # Using CoinGecko API (free tier)
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin,ethereum,ripple,litecoin,cardano,dogecoin',
            'vs_currencies': 'usd,eur,gbp,jpy'
        }
        response = requests.get(url, params=params, timeout=10)  # Added timeout
        
        if response.status_code != 200:
            raise Exception(f"API returned status code {response.status_code}")
            
        data = response.json()
        
        # Save the data locally for offline use
        with open('data/crypto_rates.json', 'w') as f:
            json.dump(data, f)
        
        # Format data to match exchange rates structure
        crypto_rates = {}
        for crypto, rates in data.items():
            crypto_key = crypto.upper()
            for currency, rate in rates.items():
                if currency.upper() == 'USD':
                    crypto_rates[crypto_key] = 1/rate  # Inverted to match exchange rate format
        
        return crypto_rates
    except Exception as e:
        # Try to load from local file if available
        try:
            with open('data/crypto_rates.json', 'r') as f:
                data = json.load(f)
            st.warning("⚠️ Using cached crypto rates (offline mode)")
            
            # Format data to match exchange rates structure
            crypto_rates = {}
            for crypto, rates in data.items():
                crypto_key = crypto.upper()
                for currency, rate in rates.items():
                    if currency.upper() == 'USD':
                        crypto_rates[crypto_key] = 1/rate
            
            return crypto_rates
        except:
            st.error(f"❌ Could not load cryptocurrency rates: {e}")
            return {}

def get_historical_rates(base_currency, target_currency, days=7):
    """Get historical exchange rate data with improved caching"""
    # Check if historical data exists locally
    filename = f'data/historical_{base_currency}_{target_currency}.json'
    
    if os.path.exists(filename):
        # Load existing data and check if it's still recent
        with open(filename, 'r') as f:
            historical_data = json.load(f)
        
        last_date = datetime.strptime(list(historical_data.keys())[-1], "%Y-%m-%d").date()
        today = datetime.now().date()
        
        if (today - last_date).days <= 1:
            return historical_data
    
    # Generate simulated historical data
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    
    # Get current rate for realism
    current_rate = 1.0
    if base_currency != target_currency:
        try:
            rates = get_exchange_rates(base_currency)
            current_rate = rates.get(target_currency, 1.0)
        except:
            current_rate = 1.0
    
    # Generate more realistic variations around the current rate
    np.random.seed(hash(f"{base_currency}{target_currency}") % 10000)
    variations = np.random.normal(0, 0.01, days)
    
    historical_rates = {}
    for i, date in enumerate(dates):
        # Make the simulated data somewhat realistic with a slight trend
        adjustment = variations[i] + (i/days) * 0.02  # Add a slight trend
        historical_rates[date] = round(current_rate * (1 + adjustment), 4)
    
    # Save data locally
    with open(filename, 'w') as f:
        json.dump(historical_rates, f)
    
    return historical_rates

def save_conversion_history(from_currency, to_currency, amount, result, timestamp=None):
    """Save conversion history to file with validation"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    history_file = 'data/conversion_history.csv'
    
    # Create history file if it doesn't exist
    if not os.path.exists(history_file):
        pd.DataFrame(columns=['timestamp', 'from_currency', 'to_currency', 'amount', 'result']).to_csv(history_file, index=False)
    
    # Read existing history
    history_df = pd.read_csv(history_file)
    
    # Add new entry
    new_entry = pd.DataFrame({
        'timestamp': [timestamp],
        'from_currency': [from_currency],
        'to_currency': [to_currency],
        'amount': [amount],
        'result': [result]
    })
    
    # Append to history and save - fixing the FutureWarning
    if history_df.empty:
        updated_df = new_entry
    else:
        updated_df = pd.concat([history_df, new_entry], ignore_index=True)
    
    updated_df.to_csv(history_file, index=False)
    return True

def load_conversion_history(limit=10):
    """Load conversion history from file with error handling"""
    history_file = 'data/conversion_history.csv'
    
    if not os.path.exists(history_file):
        return pd.DataFrame(columns=['timestamp', 'from_currency', 'to_currency', 'amount', 'result'])
    
    try:
        history_df = pd.read_csv(history_file)
        return history_df.tail(limit)
    except Exception as e:
        st.error(f"Error loading conversion history: {e}")
        return pd.DataFrame(columns=['timestamp', 'from_currency', 'to_currency', 'amount', 'result'])  
def save_user_data():
    """Save user data to a local file"""
    pass  # Add implementation here if needed
    user_data = {
        "alerts": st.session_state.alerts,
        "favorites": st.session_state.favorites,
        "conversion_history": st.session_state.conversion_history,
        "settings": {
            "default_from": st.session_state.default_from,
            "default_to": st.session_state.default_to,
            "favorite_currencies": st.session_state.favorite_currencies,
            "dark_mode": st.session_state.dark_mode
        }
    }
    
    # Save to JSON file
    with open("data/user_data.json", "w") as f:
        json.dump(user_data, f)

def load_user_data():
    """Load user data from local file"""
    try:
        if os.path.exists("data/user_data.json"):
            with open("data/user_data.json", "r") as f:
                user_data = json.load(f)
            
            # Update session state
            if "alerts" in user_data:
                st.session_state.alerts = user_data["alerts"]
            if "favorites" in user_data:
                st.session_state.favorites = user_data["favorites"]
            if "conversion_history" in user_data:
                st.session_state.conversion_history = user_data["conversion_history"]
            if "settings" in user_data:
                settings = user_data["settings"]
                if "default_from" in settings:
                    st.session_state.default_from = settings["default_from"]
                if "default_to" in settings:
                    st.session_state.default_to = settings["default_to"]
                if "favorite_currencies" in settings:
                    st.session_state.favorite_currencies = settings["favorite_currencies"]
                if "dark_mode" in settings:
                    st.session_state.dark_mode = settings["dark_mode"]
    except Exception as e:
        st.error(f"Error loading user data: {e}")

# Load user data at startup
load_user_data()
# App Header with improved styling
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 2.5rem;">💱 MultiCurrencyMatrix</h1>
    <p style="font-size: 1.2rem; color: #6c757d;">Quick, easy, and comprehensive currency conversion tools</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with improved organization
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Organized in collapsible sections for better usability
    with st.expander("🎨 Appearance", expanded=False):
        # Dark/Light mode toggle with better labeling
        dark_mode = st.toggle("Enable Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            save_user_data()
            st.rerun()
    
    with st.expander("🔄 Data Settings", expanded=True):
        # Auto-refresh settings with clearer options
        st.subheader("Auto-Refresh Data")
        refresh_interval = st.selectbox(
            "Update Frequency",
            ["Manual Only", "Every 1 minute", "Every 5 minutes", "Every 15 minutes", "Every hour"],
            index=0,
            help="Choose how often to automatically update currency rates"
        )
        
        # Refresh button with loading indicator
        if st.button("🔄 Refresh Now", use_container_width=True):
            with st.spinner("Fetching latest rates..."):
                st.session_state.rates = get_exchange_rates()
                st.session_state.crypto_rates = get_crypto_rates()
                st.session_state.last_refresh = datetime.now()
                st.success("✅ Rates updated successfully!")
        
        # Show last update time with better formatting
        last_update = st.session_state.last_refresh.strftime("%b %d, %Y at %H:%M:%S")
        st.info(f"📅 Last updated: {last_update}")
    
    # Check if it's time to refresh based on interval
    current_time = datetime.now()
    interval_seconds = 0
    
    if refresh_interval == "Every 1 minute":
        interval_seconds = 60
    elif refresh_interval == "Every 5 minutes":
        interval_seconds = 300
    elif refresh_interval == "Every 15 minutes":
        interval_seconds = 900
    elif refresh_interval == "Every hour":
        interval_seconds = 3600
    
    if interval_seconds > 0 and (current_time - st.session_state.last_refresh).seconds >= interval_seconds:
        with st.spinner("Auto-refreshing rates..."):
            st.session_state.rates = get_exchange_rates()
            st.session_state.crypto_rates = get_crypto_rates()
            st.session_state.last_refresh = current_time
    
    with st.expander("⭐ Favorite Currencies", expanded=True):
        # Available currencies with better grouping
        fiat_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "HKD", "NZD", 
                          "SEK", "KRW", "SGD", "NOK", "MXN", "INR", "RUB", "ZAR", "BRL", "TRY"]
        crypto_currencies = ["BTC", "ETH", "XRP", "LTC", "ADA", "DOGE"]
        
        # More intuitive selection with tabs
        currency_tab1, currency_tab2 = st.tabs(["Fiat", "Crypto"])
        
        with currency_tab1:
            fiat_favorites = st.multiselect(
                "Select fiat currencies",
                fiat_currencies,
                [c for c in st.session_state.favorite_currencies if c in fiat_currencies]
            )
        
        with currency_tab2:
            crypto_favorites = st.multiselect(
                "Select cryptocurrencies",
                crypto_currencies,
                [c for c in st.session_state.favorite_currencies if c in crypto_currencies]
            )
        
        # Combine selections
        selected_favorites = fiat_favorites + crypto_favorites
        
        if selected_favorites != st.session_state.favorite_currencies:
            st.session_state.favorite_currencies = selected_favorites
            save_user_data()
    
    with st.expander("🔔 Currency Alerts", expanded=False):
        # Currency alerts with better UI
        st.markdown("### Set Rate Alert")
        
        # Improved alert form
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            alert_base = st.selectbox("From", 
                                     st.session_state.favorite_currencies, 
                                     index=0 if st.session_state.favorite_currencies else None, 
                                     key="alert_base")
        
        with alert_col2:
            alert_target = st.selectbox("To", 
                                       [c for c in st.session_state.favorite_currencies if c != alert_base],
                                       index=0 if len([c for c in st.session_state.favorite_currencies if c != alert_base]) > 0 else None, 
                                       key="alert_target")
        
        alert_condition = st.selectbox("Notify me when rate is:", ["Above", "Below"], key="alert_condition")
        alert_threshold = st.number_input("Target Rate", min_value=0.001, value=1.0, step=0.001, key="alert_threshold")
        
        if st.button("Add Alert", use_container_width=True):
            st.session_state.alerts.append({
                "base": alert_base,
                "target": alert_target,
                "threshold": alert_threshold,
                "condition": alert_condition.lower(),
                "active": True
            })
            save_user_data()
            st.success(f"✅ Alert added: {alert_base}/{alert_target} {alert_condition.lower()} {alert_threshold}")
        
        if st.session_state.alerts:
            st.markdown("### Active Alerts")
            for i, alert in enumerate(st.session_state.alerts):
                if alert["active"]:
                    alert_col1, alert_col2 = st.columns([3, 1])
                    with alert_col1:
                        st.markdown(f"""
                        <div class="alert-box alert-info">
                            <strong>{alert['base']}/{alert['target']}</strong> {alert['condition']} {alert['threshold']}
                        </div>
                        """, unsafe_allow_html=True)
                    with alert_col2:
                        if st.button("Delete", key=f"remove_alert_{i}"):
                            st.session_state.alerts[i]["active"] = False
                            save_user_data()
                            st.rerun()
  # Create main tabs with better styling
tabs = st.tabs(["💱 Quick Convert", "📜 History", "📊 Charts", "🔄 Multi-Compare"])

# Fetch rates if we don't have them yet
if not st.session_state.rates:
    with st.spinner("Initializing exchange rates..."):
        st.session_state.rates = get_exchange_rates()
        st.session_state.crypto_rates = get_crypto_rates()
        st.session_state.last_refresh = datetime.now()

# Combine regular currency rates and crypto rates
all_rates = {**st.session_state.rates, **st.session_state.crypto_rates}

# Quick Convert Tab
with tabs[0]:
    st.markdown("### Quick Currency Conversion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_currency = st.selectbox(
            "From Currency",
            options=list(all_rates.keys()),
            index=list(all_rates.keys()).index(st.session_state.default_from) if st.session_state.default_from in all_rates else 0
        )
        amount = st.number_input("Amount", value=1.0, min_value=0.0, step=0.01)
    
    with col2:
        to_currency = st.selectbox(
            "To Currency",
            options=list(all_rates.keys()),
            index=list(all_rates.keys()).index(st.session_state.default_to) if st.session_state.default_to in all_rates else 1
        )
        if st.button("↔️ Swap Currencies", use_container_width=True):
            from_currency, to_currency = to_currency, from_currency
            st.session_state.default_from, st.session_state.default_to = from_currency, to_currency
            save_user_data()
            st.rerun()
    
    # Calculate conversion
    if from_currency == to_currency:
        exchange_rate = 1.0
    else:
        exchange_rate = all_rates.get(to_currency, 1.0) / all_rates.get(from_currency, 1.0)
    
    converted_amount = amount * exchange_rate
    
    # Display result
    st.markdown(f"""
    <div class="result-display">
        {amount:.2f} {from_currency} = {converted_amount:.2f} {to_currency}
    </div>
    <div style="text-align: center; margin-top: 5px;">
        1 {from_currency} = {exchange_rate:.6f} {to_currency}
    </div>
    """, unsafe_allow_html=True)
    
    # Save to history
    if st.button("Convert", use_container_width=True):
        save_conversion_history(from_currency, to_currency, amount, converted_amount)
        st.session_state.conversion_history = load_conversion_history().to_dict('records')
        save_user_data()
        st.success("Conversion saved to history!")
    
    # Add to favorites
    if st.button("⭐ Add to Favorites", use_container_width=True):
        favorite = {
            "from": from_currency,
            "to": to_currency,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        if favorite not in st.session_state.favorites:
            st.session_state.favorites.append(favorite)
            save_user_data()
            st.success("Added to favorites!")
    
    # Historical chart
    st.markdown("### Historical Trend (7 days)")
    historical_data = get_historical_rates(from_currency, to_currency)
    
    if historical_data:
        dates = list(historical_data.keys())
        rates = list(historical_data.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=rates,
            mode='lines+markers',
            name=f'{from_currency}/{to_currency}',
            line=dict(color='#4361ee', width=2)
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title='Date',
            yaxis_title='Exchange Rate',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical data not available")
        # History Tab
with tabs[1]:
    st.markdown("### Conversion History")
    
    history_df = load_conversion_history(20)
    
    if not history_df.empty:
        # Display as a nice table with formatting
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['date'] = history_df['timestamp'].dt.strftime('%Y-%m-%d')
        history_df['time'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
        
        # Format amounts nicely
        history_df['amount_str'] = history_df.apply(lambda x: f"{x['amount']:.2f} {x['from_currency']}", axis=1)
        history_df['result_str'] = history_df.apply(lambda x: f"{x['result']:.2f} {x['to_currency']}", axis=1)
        
        # Display only the columns we want
        display_df = history_df[['date', 'time', 'amount_str', 'result_str']]
        display_df.columns = ['Date', 'Time', 'From', 'To']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Time": st.column_config.TextColumn("Time"),
                "From": st.column_config.TextColumn("From"),
                "To": st.column_config.TextColumn("To")
            }
        )
        
        # Option to clear history
        if st.button("Clear History", type="secondary", use_container_width=True):
            if os.path.exists('data/conversion_history.csv'):
                os.remove('data/conversion_history.csv')
            st.session_state.conversion_history = []
            save_user_data()
            st.rerun()
    else:
        st.info("No conversion history yet. Perform conversions to see them here.")
# Charts Tab
with tabs[2]:
    st.markdown("### Currency Charts and Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_base = st.selectbox(
            "Base Currency",
            options=list(all_rates.keys()),
            index=list(all_rates.keys()).index('USD') if 'USD' in all_rates else 0,
            key="chart_base"
        )
    
    with col2:
        chart_targets = st.multiselect(
            "Compare To",
            options=list(all_rates.keys()),
            default=['EUR', 'JPY'] if all(c in all_rates for c in ['EUR', 'JPY']) else [],
            key="chart_targets"
        ) 
    
    time_period = st.selectbox(
        "Time Period",
        options=["7 days", "14 days", "30 days", "90 days", "180 days"],
        index=2,
        key="time_period"
    )
    
    days = int(time_period.split()[0])
    
    if chart_targets:
        # Get historical data for all selected currencies
        all_data = {}
        for target in chart_targets:
            historical_data = get_historical_rates(chart_base, target, days)
            if historical_data:
                all_data[target] = historical_data
        
        if all_data:
            # Prepare data for plotting
            fig = go.Figure()
            
            for target, rates in all_data.items():
                dates = list(rates.keys())
                values = list(rates.values())
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    name=f'{chart_base}/{target}',
                    hoverinfo='x+y+name'
                ))
            
            fig.update_layout(
                title=f'Exchange Rates for {chart_base}',
                xaxis_title='Date',
                yaxis_title='Exchange Rate',
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            st.markdown("### Statistics")
            
            stats_data = []
            for target, rates in all_data.items():
                values = list(rates.values())
                if values:
                    latest_rate = values[-1]
                    min_rate = min(values)
                    max_rate = max(values)
                    change = ((latest_rate - values[0]) / values[0]) * 100
                    
                    stats_data.append({
                        "Currency": target,
                        "Current Rate": latest_rate,
                        "Min": min_rate,
                        "Max": max_rate,
                        "Change (%)": change
                    })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Currency": st.column_config.TextColumn("Currency"),
                    "Current Rate": st.column_config.NumberColumn("Current Rate", format="%.4f"),
                    "Min": st.column_config.NumberColumn("Min", format="%.4f"),
                    "Max": st.column_config.NumberColumn("Max", format="%.4f"),
                    "Change (%)": st.column_config.NumberColumn("Change (%)", format="%.2f%%")
                }
            )
        else:
            st.warning("No historical data available for the selected currencies")
    else:
        st.info("Select at least one target currency to compare")

 # Multi-Compare Tab
# Multi-Compare Tab
with tabs[3]:
    st.markdown("### Multi-Currency Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_amount = st.number_input("Amount", value=1.0, min_value=0.0, step=0.01, key="compare_amount")
        compare_base = st.selectbox(
            "From Currency",
            options=list(all_rates.keys()),
            index=list(all_rates.keys()).index('USD') if 'USD' in all_rates else 0,
            key="compare_base"
        )
    
    with col2:
        # Get available currencies
        available_currencies = list(all_rates.keys())
        
        # Filter favorite currencies to only include those that exist in all_rates
        valid_favorites = [c for c in st.session_state.favorite_currencies if c in available_currencies]
        
        # Set safe defaults (EUR, GBP, JPY) only if they exist in available currencies
        safe_defaults = [c for c in ['EUR', 'GBP', 'JPY'] if c in available_currencies]
        
        # Use valid favorites if they exist, otherwise use safe defaults
        default_targets = valid_favorites if valid_favorites else safe_defaults
        
        compare_targets = st.multiselect(
            "To Currencies",
            options=available_currencies,
            default=default_targets[:3],  # Limit to first 3 to avoid clutter
            key="compare_targets"
        )
    
    if compare_targets:
        # Rest of the code remains the same...
        # Calculate conversions
        results = []
        for target in compare_targets:
            if compare_base == target:
                rate = 1.0
            else:
                rate = all_rates.get(target, 1.0) / all_rates.get(compare_base, 1.0)
            
            converted = compare_amount * rate
            results.append({
                "Currency": target,
                "Exchange Rate": rate,
                "Converted Amount": converted
            })
        
        results_df = pd.DataFrame(results)
        
        # Display results
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Currency": st.column_config.TextColumn("Currency"),
                "Exchange Rate": st.column_config.NumberColumn("Exchange Rate", format="%.6f"),
                "Converted Amount": st.column_config.NumberColumn("Converted Amount", format="%.2f")
            }
        )
        
        # Option to save as CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"currency_comparison_{compare_base}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Select at least one target currency to compare")
def main():
    # Check for alerts
    for alert in st.session_state.alerts:
        if alert["active"]:
            base = alert["base"]
            target = alert["target"]
            threshold = alert["threshold"]
            condition = alert["condition"]
            
            if base in all_rates and target in all_rates:
                current_rate = all_rates[target] / all_rates[base]
                
                if (condition == "above" and current_rate > threshold) or \
                   (condition == "below" and current_rate < threshold):
                    st.toast(f"🚨 Alert: {base}/{target} is {current_rate:.4f} ({condition} {threshold})", icon="⚠️")

if dark_mode_wrapper_open:
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()                                               