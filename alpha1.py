import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hurst import compute_Hc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from pypfopt import expected_returns, risk_models, EfficientFrontier
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# Set page configuration
st.set_page_config(
    page_title="UTY CAPITAL - US Equity Analysis Tool 01",
    layout="wide"
)

# Define password
CORRECT_PASSWORD = "utycapital"  # You can change this to your desired password

# Initialize session state for password validation
if 'password_correct' not in st.session_state:
    st.session_state.password_correct = False

# Title and description
st.title("UTY CAPITAL - US Equity Analysis Tool 01")

# Password protection section
if not st.session_state.password_correct:
    st.markdown("Please enter the password to access the application.")
    password_input = st.text_input("Password", type="password")
    if st.button("Submit"):
        if password_input == CORRECT_PASSWORD:
            st.session_state.password_correct = True
            st.success("Password correct! Access granted.")
            #st.experimental_rerun()  # Rerun the app to show the content
        else:
            st.error("Incorrect password. Please try again.")
    
    # Stop execution here if password is not correct
    st.stop()

# If password is correct, continue with the main application
st.markdown("Analyze US stocks with technical indicators including 50 EMA, Hurst Exponent, Volatility Regime, Bullish/Bearish Regime, and Portfolio Optimization")

# FIXED: StockRiskAnalyzer class with proper scalar handling
class StockRiskAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.fetch_data()
        self.calculate_log_returns()
    
    def fetch_data(self, period="1y"):
        # Fetch stock data
        data = yf.download(self.ticker, period=period)
        return data
    
    def calculate_log_returns(self):
        # Calculate log returns
        self.data['LogReturn'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data = self.data.dropna()

    def calculate_var_stop_loss(self, 
                                 holding_days: int, 
                                 position_type: str, 
                                 confidence_level: float = 0.95):
        """
        Calculate Value at Risk (VaR) and Stop Loss price.
        """
        if self.data is None or len(self.data) < 2:
            raise ValueError("Not enough historical data to calculate returns.")

        # Calculate daily volatility - ensure it's a scalar
        daily_volatility = float(self.data['LogReturn'].std())
        
        # Calculate Z-score
        z_score = norm.ppf(confidence_level)
        
        # Calculate VaR percentage
        var_percent = z_score * daily_volatility * np.sqrt(holding_days)
        
        # Get latest closing price - ensure it's a scalar
        entry_price = float(self.data['Close'].iloc[-1])
        
        # Calculate stop loss based on position type
        if position_type.lower() == 'long':
            stop_loss_price = entry_price * (1 - var_percent)
        elif position_type.lower() == 'short':
            stop_loss_price = entry_price * (1 + var_percent)
        else:
            raise ValueError("Position type must be 'long' or 'short'.")
        
        return {
            "Stock": self.ticker,
            "Position": position_type,
            "Holding Days": holding_days,
            "Confidence Level": confidence_level,
            "VaR %": round(var_percent * 100, 2),
            "Entry Price": round(entry_price, 2),
            "Stop Loss Price": round(stop_loss_price, 2)
        }

    def calculate_target_price(self, 
                                stop_loss_price: float, 
                                position_type: str, 
                                reward_ratio: float = 2.0):
        """
        Calculate target price based on risk-reward ratio.
        """
        # Get latest closing price as entry price - ensure it's a scalar
        entry_price = float(self.data['Close'].iloc[-1])

        if position_type.lower() == 'long':
            risk = entry_price - stop_loss_price
            target_price = entry_price + reward_ratio * risk
        elif position_type.lower() == 'short':
            risk = stop_loss_price - entry_price
            target_price = entry_price - reward_ratio * risk
        else:
            raise ValueError("Position type must be 'long' or 'short'.")
        
        return {
            "Entry Price": round(entry_price, 2),
            "Stop Loss Price": round(stop_loss_price, 2),
            "Risk per Unit": round(risk, 2),
            "Reward Ratio": reward_ratio,
            "Target Price": round(target_price, 2)
        }

# Function to get all US tickers
@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def get_all_tickers():
    try:
        # Try to get a predefined list of common US tickers
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        tickers_df = pd.read_csv(url, header=None)
        tickers = sorted(tickers_df[0].tolist())
        return tickers
    except:
        # Fallback to a smaller predefined list
        return sorted([
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
            "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "INTC", "VZ", "DIS", "ADBE",
            "CSCO", "CRM", "NFLX", "PFE", "KO", "PEP", "T", "MRK", "CMCSA"
        ])

# Function to verify if ticker exists
def verify_ticker(ticker):
    try:
        data = yf.download(ticker, period="1d", progress=False)
        return not data.empty
    except:
        return False

# Modified Hurst Exponent calculation
def hurst_exponent(prices):
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    try:
        H, c, data_reg = compute_Hc(prices, kind='price', simplified=True)
        return H
    except Exception as e:
        st.error(f"Hurst calculation error: {str(e)}")
        return 0.5

# Function to interpret Hurst Exponent
def interpret_hurst(hurst_value):
    if hurst_value < 0.5:
        return "Mean Reversion"
    elif hurst_value > 0.5:
        return "Trending Behavior"
    else:
        return "Random Walk"

# Function to calculate 50 EMA status
def ema_status(data):
    current_price = float(data['Close'].iloc[-1])
    ema_50 = float(data['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
    return "Above" if current_price > ema_50 else "Below", ema_50

# Function to determine volatility regime
def volatility_regime(data):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.rolling(window=20).std().dropna()
    vol_threshold = volatility.median()
    
    # Extract the last value as a scalar
    current_vol = float(volatility.iloc[-1])
    vol_threshold_value = float(vol_threshold)
    
    regime = "Low Volatility" if current_vol < vol_threshold_value else "High Volatility"
    
    return regime, current_vol, vol_threshold_value

# Function to determine bullish/bearish regime using HMM
def hmm_regime(data):
    # Prepare features
    df = pd.DataFrame()
    df['Close'] = data['Close']
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Volume'] = data['Volume']
    df['Volume_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)
    
    if len(df) < 30:  # Need sufficient data for HMM
        return "Insufficient Data", None, None
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Return', 'Volatility', 'Volume_Change']])
    
    # Train HMM with multiple seeds to find best model
    best_score = -np.inf
    best_model = None
    best_states = None
    
    for seed in range(5):  # Reduced from 30 to 5 seeds for efficiency
        model = GaussianHMM(n_components=2, covariance_type="tied", n_iter=1000, random_state=seed)
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
                best_states = model.predict(X)
        except:
            continue
    
    if best_model is None:
        return "Model Failed", None, None
    
    # Label states as Bull or Bear based on avg return
    df['Regime'] = best_states
    regime_returns = df.groupby('Regime')['Return'].mean()
    bull_state = regime_returns.idxmax()
    bear_state = regime_returns.idxmin()
    last_state = df['Regime'].iloc[-1]
    
    regime = "Bullish" if last_state == bull_state else "Bearish"
    avg_return = regime_returns[last_state]
    
    return regime, avg_return, df['Regime']

# Function to download stock data and compute indicators
def analyze_ticker(ticker, period="1y"):
    try:
        # Download data
        data = yf.download(ticker, period=period)
        
        if len(data) < 50:
            return {"ticker": ticker, "error": "Insufficient data"}
        
        # Calculate 50 EMA status
        ema_position, ema_value = ema_status(data)
        
        # Calculate Hurst Exponent
        hurst_value = hurst_exponent(data['Close'])
        hurst_interpretation = interpret_hurst(hurst_value)
        
        # Calculate Volatility Regime
        vol_regime, current_vol, vol_threshold = volatility_regime(data)
        
        # Calculate Bullish/Bearish Regime
        bull_bear_regime, avg_return, regime_series = hmm_regime(data)
        
        # Current price - convert to scalar
        current_price = float(data['Close'].iloc[-1])
        
        # Get company name
        try:
            ticker_info = yf.Ticker(ticker).info
            company_name = ticker_info.get('shortName', ticker)
        except:
            company_name = ticker
        
        result = {
            "ticker": ticker,
            "name": company_name,
            "current_price": current_price,
            "50_ema": round(ema_value, 2),
            "50_ema_status": ema_position,
            "hurst_exponent": round(hurst_value, 3),
            "hurst_interpretation": hurst_interpretation,
            "volatility_regime": vol_regime,
            "current_volatility": round(current_vol, 4),
            "vol_threshold": round(vol_threshold, 4),
            "bull_bear_regime": bull_bear_regime,
            "avg_regime_return": None if avg_return is None else round(avg_return, 4),
            "regime_series": regime_series,
            "data": data
        }
        
        return result
    except Exception as e:
        import traceback
        error_message = f"{str(e)}\n{traceback.format_exc()}"
        return {"ticker": ticker, "error": error_message}

# Function to perform correlation analysis and find maximally diversified subset
def analyze_correlations(tickers, period="1y"):
    try:
        # Download close prices
        end_date = datetime.now()
        
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
            
        # Download with explicit start and end dates
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Build graph for diversification analysis
        threshold = 0.3  # Fixed threshold for consistency
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(tickers)
        
        # Add edges for stocks with correlation below threshold
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                stock1 = tickers[i]
                stock2 = tickers[j]
                if stock1 in correlation_matrix.index and stock2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[stock1, stock2]
                    if abs(corr) <= threshold:
                        G.add_edge(stock1, stock2)
        
        # Find cliques
        cliques = list(nx.find_cliques(G))
        largest_clique = max(cliques, key=len) if cliques else []
        
        # Create network graph figure
        plt.figure(figsize=(12, 8))
        
        pos = nx.spring_layout(G, seed=42)
        
        node_colors = ['lightgreen' if node in largest_clique else 'skyblue' for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title("Low Correlation Stock Network (Edges: Correlation ≤ 0.3)", fontsize=16)
        
        return {
            "correlation_matrix": correlation_matrix,
            "diversified_subset": largest_clique,
            "network_graph": plt,
            "graph_object": G,
            "price_data": data  # Add price data for portfolio optimization
        }
    except Exception as e:
        st.error(f"Correlation analysis error: {str(e)}")
        return None

# Portfolio optimization to calculate weights
def optimize_portfolio(price_data, allow_shorting=False):
    try:
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
        
        # Create the Efficient Frontier object
        weight_bounds = (-1, 1) if allow_shorting else (0, 1)
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        
        # Maximize Sharpe Ratio
        ef.max_sharpe()
        
        # Get cleaned weights
        cleaned_weights = ef.clean_weights()
        
        # Get portfolio performance
        expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        
        return {
            "weights": cleaned_weights,
            "expected_annual_return": expected_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio
        }
    except Exception as e:
        st.error(f"Portfolio optimization error: {str(e)}")
        return None

# FIXED: Function to perform SL-TP Analysis for given tickers
def perform_sltp_analysis(tickers, holding_days=10, position_type='long', reward_ratio=2.0):
    # Container to store analysis results for all tickers
    sltp_results = []
    
    for ticker in tickers:
        try:
            # Create analyzer instance
            analyzer = StockRiskAnalyzer(ticker)
            
            # Calculate VaR and Stop Loss
            var_result = analyzer.calculate_var_stop_loss(
                holding_days=holding_days, 
                position_type=position_type
            )
            
            # Calculate Target Price
            target_result = analyzer.calculate_target_price(
                stop_loss_price=var_result['Stop Loss Price'], 
                position_type=position_type,
                reward_ratio=reward_ratio
            )
            
            # Combine results
            combined_result = {**var_result}
            
            # Ensure we only add non-duplicate fields from target_result
            for key, value in target_result.items():
                if key not in combined_result:
                    combined_result[key] = value
            
            sltp_results.append(combined_result)
        
        except Exception as e:
            # If analysis fails for a ticker, add an error entry
            sltp_results.append({
                "Stock": ticker,
                "Error": str(e)
            })
    
    return sltp_results
    
# Get all tickers
all_tickers = get_all_tickers()

# Sidebar for filtering and selections
st.sidebar.header("Enter Stock Tickers & Settings")

# Period selection
period = st.sidebar.selectbox(
    "Select Time Period for Analysis",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3  # Default to 1 year
)

# Correlation threshold slider
correlation_threshold = st.sidebar.slider(
    "Correlation Threshold for Diversification",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
    help="Stocks with correlation below this threshold are considered diversified"
)

# Add shorting option for portfolio optimization
allow_shorting = st.sidebar.checkbox(
    "Enable Short Selling in Portfolio Optimization",
    value=False,
    help="When enabled, the optimizer can recommend negative weights for some stocks"
)

# Comma separated input for tickers
ticker_input = st.sidebar.text_area(
    "Enter Tickers (comma-separated)",
    "AAPL, MSFT, AMZN",
    help="Example: AAPL, MSFT, AMZN, GOOGL"
)

# Process the comma-separated input
if ticker_input:
    # Split by comma, strip whitespace, convert to uppercase
    input_tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    
    # Create a container to display validation results
    validation_container = st.sidebar.container()
    
    # Display the number of entered tickers
    validation_container.write(f"Entered {len(input_tickers)} tickers")
    
    # Add Validate button
    validate_button = st.sidebar.button("Validate Tickers")
    
    # Add SL-TP Analysis section
    st.sidebar.markdown("---")  # Separator
    st.sidebar.subheader("Stop Loss - Target Price Analysis")
    
    # Holding Days input
    holding_days = st.sidebar.number_input(
        "Holding Days", 
        min_value=1, 
        max_value=365, 
        value=10,
        help="Number of days to hold the position"
    )
    
    # Position Type selection
    position_type = st.sidebar.selectbox(
        "Position Type",
        ["Long", "Short"],
        help="Select whether you're going long or short on the stocks"
    )
    
    # Reward Ratio input
    reward_ratio = st.sidebar.number_input(
        "Reward-Risk Ratio", 
        min_value=1.0, 
        max_value=10.0, 
        value=2.0,
        step=0.5,
        help="Desired reward-to-risk ratio for target price calculation"
    )
    
    # SL-TP Analysis button
    sltp_button = st.sidebar.button("SL-TP Analysis")
    
    if validate_button:
        # Container for validation results
        with validation_container:
            st.write("Validating tickers...")
            
            # Validate each ticker
            valid_tickers = []
            invalid_tickers = []
            
            with st.spinner("Checking tickers..."):
                for ticker in input_tickers:
                    if verify_ticker(ticker):
                        valid_tickers.append(ticker)
                    else:
                        invalid_tickers.append(ticker)
            
            # Display validation results
            if valid_tickers:
                st.success(f"✅ Valid tickers: {', '.join(valid_tickers)}")
            
            if invalid_tickers:
                st.error(f"❌ Invalid tickers not found: {', '.join(invalid_tickers)}")
            
            # Update selected tickers to only include valid ones
            selected_tickers = valid_tickers
    else:
        # Use all entered tickers if not validated
        selected_tickers = input_tickers
else:
    selected_tickers = []

# Analyze button
analyze_button = st.sidebar.button("Analyze Selected Stocks")

# Handle SL-TP Analysis button click
if sltp_button and selected_tickers:
    st.header("Stop Loss - Target Price Analysis")
    
    with st.spinner("Performing SL-TP Analysis..."):
        # Perform analysis
        sltp_results = perform_sltp_analysis(
            selected_tickers, 
            holding_days=holding_days, 
            position_type=position_type.lower(), 
            reward_ratio=reward_ratio
        )
        
        # Display results
        if sltp_results:
            # Create DataFrame for display
            display_df = pd.DataFrame(sltp_results)
            
            # Handle potential errors
            if 'Error' in display_df.columns:
                # Separate successful and error results
                error_rows = display_df[display_df['Error'].notna()]
                success_rows = display_df[display_df['Error'].isna()].drop(columns=['Error'])
                
                # Display success results
                if not success_rows.empty:
                    st.subheader("SL-TP Analysis Results")
                    st.dataframe(success_rows, use_container_width=True)
                
                # Display error results if any
                if not error_rows.empty:
                    st.error("Errors in Analysis:")
                    st.dataframe(error_rows, use_container_width=True)
            else:
                # No errors - display full results
                st.subheader("SL-TP Analysis Results")
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No results generated. Please check the tickers and analysis parameters.")

if analyze_button and selected_tickers:
    # First verify all tickers before analysis
    st.sidebar.write("Verifying tickers before analysis...")
    
    valid_for_analysis = []
    invalid_for_analysis = []
    
    with st.spinner("Verifying tickers..."):
        for ticker in selected_tickers:
            if verify_ticker(ticker):
                valid_for_analysis.append(ticker)
            else:
                invalid_for_analysis.append(ticker)
    
    # Display validation results
    if invalid_for_analysis:
        st.error(f"❌ The following tickers were not found and will be skipped: {', '.join(invalid_for_analysis)}")
    
    if not valid_for_analysis:
        st.error("No valid tickers to analyze. Please enter valid ticker symbols.")
    else:
        st.success(f"Proceeding with analysis of {len(valid_for_analysis)} valid tickers: {', '.join(valid_for_analysis)}")
        
        with st.spinner("Analyzing selected stocks..."):
            # First, perform correlation analysis if multiple tickers
            if len(valid_for_analysis) > 1:
                st.header("Correlation Analysis")
                
                with st.spinner("Calculating correlations and diversification..."):
                    correlation_results = analyze_correlations(valid_for_analysis, period)
                    
                    if correlation_results:
                        # Display correlation heatmap
                        st.subheader("Stock Return Correlation Heatmap")
                        
                        # Convert correlation matrix to plotly heatmap
                        fig = px.imshow(
                            correlation_results["correlation_matrix"],
                            x=correlation_results["correlation_matrix"].columns,
                            y=correlation_results["correlation_matrix"].index,
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            text_auto=".2f",
                            title="Stock Return Correlations"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display maximally diversified subset
                        st.subheader("Maximally Diversified Subset")
                        diversified_subset = correlation_results["diversified_subset"]
                        
                        if diversified_subset:
                            st.write(f"The following {len(diversified_subset)} stocks form a maximally diversified subset (mutual correlation ≤ {correlation_threshold}):")
                            st.write(", ".join(sorted(diversified_subset)))
                            
                            # Add some additional diversification metrics
                            if len(diversified_subset) > 1:
                                # Calculate average correlation within diversified subset
                                subset_corr = correlation_results["correlation_matrix"].loc[diversified_subset, diversified_subset]
                                avg_corr = (subset_corr.sum().sum() - len(diversified_subset)) / (len(diversified_subset) * (len(diversified_subset) - 1))
                                st.write(f"Average correlation within diversified subset: {avg_corr:.4f}")
                        else:
                            st.write(f"No maximally diversified subset found with the correlation threshold of {correlation_threshold}.")
                        
                        # Display network graph
                        st.subheader("Low Correlation Network Visualization")
                        st.write("Nodes connected by edges have correlations ≤ 0.3 with each other.")
                        st.write("Green nodes represent stocks in the maximally diversified subset.")
                        st.pyplot(correlation_results["network_graph"].gcf())
                        
                        # Portfolio optimization section
                        st.header("Portfolio Optimization")
                        with st.spinner("Calculating optimal portfolio weights..."):
                            # Clean the price data (drop columns with NaN values)
                            price_data = correlation_results["price_data"].dropna(axis=1)
                            
                            if price_data.shape[1] > 1:  # Need at least 2 stocks for optimization
                                optimization_results = optimize_portfolio(price_data, allow_shorting)
                                
                                if optimization_results:
                                    st.subheader("Maximum Sharpe Ratio Portfolio")
                                    
                                    # Format the portfolio performance metrics
                                    st.write(f"Expected Annual Return: {optimization_results['expected_annual_return']:.2%}")
                                    st.write(f"Annual Volatility: {optimization_results['annual_volatility']:.2%}")
                                    st.write(f"Sharpe Ratio: {optimization_results['sharpe_ratio']:.2f}")
                                    
                                    # Convert weights to DataFrame and show
                                    weights_df = pd.DataFrame({
                                        'Stock': list(optimization_results['weights'].keys()),
                                        'Weight': list(optimization_results['weights'].values())
                                    })
                                    weights_df = weights_df.sort_values(by='Weight', ascending=False)
                                    
                                    # Format weights as percentages
                                    weights_df['Weight'] = weights_df['Weight'].map('{:.2%}'.format)
                                    
                                    st.write("Optimal Portfolio Weights:")
                                    st.dataframe(weights_df, use_container_width=True)
                                    
                                    # Create pie chart for weights
                                    weights_for_chart = {k: v for k, v in optimization_results['weights'].items() if v > 0.01}
                                    others_weight = sum(v for k, v in optimization_results['weights'].items() if v <= 0.01)
                                    
                                    if others_weight > 0:
                                        weights_for_chart['Others'] = others_weight
                                    
                                    fig = px.pie(
                                        values=list(weights_for_chart.values()),
                                        names=list(weights_for_chart.keys()),
                                        title="Portfolio Allocation",
                                        hole=0.3
                                    )
                                    fig.update_traces(textposition='inside', textinfo='percent+label')
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    if any(v < 0 for v in optimization_results['weights'].values()):
                                        st.info("Note: Negative weights indicate short positions.")
                                else:
                                    st.error("Portfolio optimization failed. This could be due to insufficient data or high correlations among stocks.")
                            else:
                                st.warning("Portfolio optimization requires at least 2 stocks with complete data.")
            
            # Run individual stock analysis on selected tickers
            st.header("Individual Stock Analysis")
            results = []
            for ticker in valid_for_analysis:
                with st.spinner(f"Analyzing {ticker}..."):
                    result = analyze_ticker(ticker, period)
                    results.append(result)
            
            # Create a DataFrame from valid results
            valid_results = [r for r in results if "error" not in r]
            error_results = [r for r in results if "error" in r]
            
            if valid_results:
                # Create DataFrame for display
                display_df = pd.DataFrame([
                    {
                        "Ticker": r["ticker"],
                        "Name": r["name"],
                        "Price": f"${r['current_price']:.2f}",
                        "50 EMA": f"${r['50_ema']:.2f}",
                         "Hurst": r["hurst_exponent"],
                        "Interpretation": r["hurst_interpretation"],
                        "Vol Regime": r["volatility_regime"],
                        "Trend Regime": r["bull_bear_regime"]
                    } 
                    for r in valid_results
                ])
                
                # Display results in a table
                st.subheader("Technical Analysis Results")
                st.dataframe(display_df, use_container_width=True)

else:
       # Initial state - instructions
    st.info("👈 Enter comma-separated ticker symbols in the sidebar, then click 'Analyze Selected Stocks'")
    
    # Basic description
    st.markdown("""
    ## How to use this tool:
    
    1. **Enter tickers** in comma-separated format (e.g., AAPL, MSFT, GOOGL) in the sidebar
    2. Choose a **time period** for analysis
    3. Adjust the **correlation threshold** for diversification analysis
    4. Choose whether to **enable short selling** for portfolio optimization
    5. Click **Validate Tickers** to check if all your tickers exist (optional)
    6. Click **Analyze Selected Stocks** to run the analysis
    7. View results in the various charts and tables
    
    ## Analysis components:
    
    - **Portfolio Optimization**: Calculates optimal portfolio weights using Modern Portfolio Theory to maximize the Sharpe ratio
    - **Correlation Analysis**: Visualizes the correlations between selected stocks and identifies a maximally diversified subset
    - **Network Visualization**: Shows which stocks have low correlation with each other (connected by edges)
    - **50 EMA Status**: Indicates if the current price is above or below the 50-day Exponential Moving Average
    - **Hurst Exponent**: A measure of long-term memory of time series that helps identify:
      - **Mean Reversion** (< 0.5): Price tends to revert to the mean
      - **Random Walk** (= 0.5): Price follows a random walk with no pattern
      - **Trending Behavior** (> 0.5): Price tends to continue trends
    - **Volatility Regime**: Identifies whether a stock is in a high or low volatility period
    - **Bull/Bear Regime**: Uses Hidden Markov Model (HMM) to detect bullish or bearish market regimes
    """)
