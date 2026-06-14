import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── DARK AIR QUALITY THEME ────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0C0F14',
    'axes.facecolor':   '#161B24',
    'axes.edgecolor':   '#262E3D',
    'axes.labelcolor':  '#CBD5E1',
    'xtick.color':      '#64748B',
    'ytick.color':      '#64748B',
    'grid.color':       '#262E3D',
    'grid.alpha':       0.6,
    'text.color':       '#E2E8F0',
    'font.family':      'DejaVu Sans',
    'axes.titlepad':    14,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
})
GREEN, YELLOW, ORANGE, RED, PURPLE, MAROON = '#4ADE80','#FACC15','#FB923C','#F87171','#C084FC','#991B1B'
BLUE = '#60A5FA'

print("✅ Imports OK")

# ── AQI CATEGORY FUNCTION (Indian CPCB-style breakpoints, simplified) ─
def pm25_to_aqi(pm25):
    # Simplified linear breakpoints based on CPCB AQI for PM2.5
    bp = [(0,30,0,50),(30,60,50,100),(60,90,100,200),(90,120,200,300),(120,250,300,400),(250,500,400,500)]
    for c_lo,c_hi,a_lo,a_hi in bp:
        if c_lo <= pm25 <= c_hi:
            return a_lo + (pm25-c_lo)/(c_hi-c_lo)*(a_hi-a_lo)
    return 500

def aqi_category(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

AQI_COLORS = {'Good':GREEN,'Satisfactory':'#A3E635','Moderate':YELLOW,
               'Poor':ORANGE,'Very Poor':RED,'Severe':MAROON}

# ── SIMULATE 7 DAYS OF AIR QUALITY DATA (HOURLY) ──────────────
np.random.seed(42)
timestamps = pd.date_range('2024-06-01', periods=7*24, freq='1h')
n = len(timestamps)
hours = timestamps.hour.to_numpy()

# PM2.5: higher during traffic hours (8-10am, 6-9pm), lower at night
base_pm25 = 45
traffic_boost = np.where(np.isin(hours, [8,9,18,19,20]), 35, 0)
night_dip = np.where(np.isin(hours, [2,3,4,5]), -15, 0)
pm25 = base_pm25 + traffic_boost + night_dip + np.random.normal(0, 8, n)

# Inject a pollution spike event (Day 4, industrial activity / stubble burning style)
spike_mask = (np.arange(n) >= 72) & (np.arange(n) < 84)
pm25[spike_mask] += 90

pm25 = np.clip(pm25, 10, 350)

# CO2 (ppm) - indoor/ambient baseline with traffic correlation
co2 = 410 + (pm25-45)*0.8 + np.random.normal(0, 10, n)
co2 = np.clip(co2, 380, 900)

# CO (ppm) - correlates with traffic
co = 0.5 + (pm25-45)*0.015 + np.random.normal(0, 0.2, n)
co = np.clip(co, 0.1, 9)

# Temperature & Humidity (daily cycle)
temperature = 27 + 6*np.sin(2*np.pi*(hours-8)/24) + np.random.normal(0,0.8,n)
humidity = 65 - (temperature-27)*1.8 + np.random.normal(0,3,n)
humidity = np.clip(humidity, 30, 90)

# AQI from PM2.5
aqi = np.array([pm25_to_aqi(v) for v in pm25]).round(0)
category = [aqi_category(a) for a in aqi]

df = pd.DataFrame({
    'timestamp': timestamps,
    'pm25': pm25.round(1),
    'co2_ppm': co2.round(0),
    'co_ppm': co.round(2),
    'temperature_c': temperature.round(1),
    'humidity_pct': humidity.round(1),
    'aqi': aqi,
    'aqi_category': category,
})
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# ── THRESHOLD ALERTS ───────────────────────────────────────────
THRESH = {'pm25_unsafe':100, 'co2_high':800, 'co_high':4.0, 'aqi_poor':200}
df['alert_pm25'] = (df['pm25'] > THRESH['pm25_unsafe']).astype(int)
df['alert_co2']  = (df['co2_ppm'] > THRESH['co2_high']).astype(int)
df['alert_co']   = (df['co_ppm'] > THRESH['co_high']).astype(int)
df['alert_aqi']  = (df['aqi'] > THRESH['aqi_poor']).astype(int)
df['any_alert']  = df[['alert_pm25','alert_co2','alert_co','alert_aqi']].max(axis=1)

df.to_csv('air_quality_data.csv', index=False)
print(f"✅ Generated {len(df)} hourly readings over 7 days")
print(f"🌫️  Avg PM2.5: {df['pm25'].mean():.1f} µg/m³")
print(f"📊 Avg AQI: {df['aqi'].mean():.0f} ({aqi_category(df['aqi'].mean())})")
print(f"⚠️  Total alerts: {df['any_alert'].sum()}")
print(f"   AQI category distribution: {dict(df['aqi_category'].value_counts())}")

# ── CHART 1 — MAIN DASHBOARD ──────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('#0C0F14')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# AQI over time, colored by category
ax1 = fig.add_subplot(gs[0,0]); ax1.set_facecolor('#161B24')
ax1.plot(df['timestamp'], df['aqi'], color=BLUE, linewidth=1.2, alpha=0.6, zorder=2)
for cat, color in AQI_COLORS.items():
    subset = df[df['aqi_category']==cat]
    if len(subset)>0:
        ax1.scatter(subset['timestamp'], subset['aqi'], color=color, s=12, label=cat, zorder=3)
ax1.axhline(y=THRESH['aqi_poor'], color=RED, linestyle='--', linewidth=1.2, alpha=0.6, label='Poor Threshold (200)')
ax1.set_title('📊  AQI Over Time (Spike on Day 4)')
ax1.set_ylabel('AQI')
ax1.legend(framealpha=0.15, labelcolor='white', fontsize=7, ncol=2)
ax1.grid(True, alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', rotation=25)

# PM2.5 with threshold
ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor('#161B24')
ax2.plot(df['timestamp'], df['pm25'], color=ORANGE, linewidth=1.5)
ax2.fill_between(df['timestamp'], df['pm25'], alpha=0.12, color=ORANGE)
ax2.axhline(y=THRESH['pm25_unsafe'], color=RED, linestyle='--', linewidth=1.2, alpha=0.7, label='Unsafe (100 µg/m³)')
ax2.set_title('🌫️  PM2.5 Concentration')
ax2.set_ylabel('µg/m³')
ax2.legend(framealpha=0.15, labelcolor='white', fontsize=8)
ax2.grid(True, alpha=0.25)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', rotation=25)

# CO2 & CO dual axis
ax3 = fig.add_subplot(gs[1,0]); ax3.set_facecolor('#161B24')
ax3.plot(df['timestamp'], df['co2_ppm'], color=PURPLE, linewidth=1.5, label='CO₂ (ppm)')
ax3.axhline(y=THRESH['co2_high'], color=RED, linestyle='--', linewidth=1, alpha=0.6)
ax3b = ax3.twinx()
ax3b.plot(df['timestamp'], df['co_ppm'], color=YELLOW, linewidth=1.2, alpha=0.8, label='CO (ppm)')
ax3b.set_ylabel('CO (ppm)', color=YELLOW)
ax3b.tick_params(axis='y', colors=YELLOW)
ax3.set_title('💨  CO₂ & CO Levels')
ax3.set_ylabel('CO₂ (ppm)', color=PURPLE)
ax3.tick_params(axis='y', colors=PURPLE)
ax3.legend(loc='upper left', framealpha=0.15, labelcolor='white', fontsize=8)
ax3.grid(True, alpha=0.25)
ax3.spines['top'].set_visible(False)
ax3.tick_params(axis='x', rotation=25)

# Temperature & Humidity
ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor('#161B24')
ax4.plot(df['timestamp'], df['temperature_c'], color=ORANGE, linewidth=1.5, label='Temp (°C)')
ax4b = ax4.twinx()
ax4b.plot(df['timestamp'], df['humidity_pct'], color=BLUE, linewidth=1.2, alpha=0.7, label='Humidity (%)')
ax4b.set_ylabel('Humidity (%)', color=BLUE)
ax4b.tick_params(axis='y', colors=BLUE)
ax4.set_title('🌡️  Temperature & Humidity')
ax4.set_ylabel('Temperature (°C)', color=ORANGE)
ax4.tick_params(axis='y', colors=ORANGE)
ax4.legend(loc='upper left', framealpha=0.15, labelcolor='white', fontsize=8)
ax4.grid(True, alpha=0.25)
ax4.spines['top'].set_visible(False)
ax4.tick_params(axis='x', rotation=25)

fig.suptitle('🌍  IoT Air Quality Monitoring Dashboard — 7-Day Sensor Data', fontsize=15, color='#E2E8F0', y=1.01)
plt.savefig('chart1_air_quality_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0C0F14')
plt.close(); print("Saved chart1")

# ── CHART 2 — AQI CATEGORY BREAKDOWN ──────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

cat_order = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
cat_counts = df['aqi_category'].value_counts().reindex(cat_order).fillna(0)
bars = ax1.bar(cat_counts.index, cat_counts.values,
               color=[AQI_COLORS[c] for c in cat_order], alpha=0.88, edgecolor='none')
ax1.set_title('🏷️  AQI Category Distribution (7 Days)')
ax1.set_ylabel('Hours')
ax1.tick_params(axis='x', rotation=20)
ax1.grid(True, axis='y', alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
for b, v in zip(bars, cat_counts.values):
    if v>0: ax1.text(b.get_x()+b.get_width()/2, v+1, f'{int(v)}', ha='center', color='#E2E8F0', fontsize=9)

# Hourly avg PM2.5 pattern (shows traffic hours)
hourly_avg = df.groupby('hour')['pm25'].mean()
colors_h = [RED if h in [8,9,18,19,20] else BLUE for h in hourly_avg.index]
ax2.bar(hourly_avg.index, hourly_avg.values, color=colors_h, alpha=0.85, edgecolor='none')
ax2.set_title('🚗  Avg PM2.5 by Hour of Day\n(Red = Traffic Hours)')
ax2.set_xlabel('Hour'); ax2.set_ylabel('PM2.5 (µg/m³)')
ax2.grid(True, axis='y', alpha=0.25)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart2_aqi_breakdown.png', dpi=150, bbox_inches='tight', facecolor='#0C0F14')
plt.close(); print("Saved chart2")

# ── CHART 3 — DAILY HEATMAP (AQI BY HOUR) ─────────────────────
pivot = df.pivot_table(values='aqi', index='date', columns='hour', aggfunc='mean')
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot, annot=False, cmap='RdYlGn_r', linewidths=0.3, linecolor='#0C0F14',
            cbar_kws={'label':'AQI'}, ax=ax, vmin=0, vmax=300)
ax.set_title('🗓️  AQI Heatmap (Day × Hour) — Spike Visible on Day 4', fontsize=12, pad=12)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Date')
plt.tight_layout()
plt.savefig('chart3_aqi_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0C0F14')
plt.close(); print("Saved chart3")

# ── CHART 4 — ALERT SUMMARY ────────────────────────────────────
alert_counts = {
    'PM2.5 Unsafe': df['alert_pm25'].sum(),
    'CO₂ High': df['alert_co2'].sum(),
    'CO High': df['alert_co'].sum(),
    'AQI Poor+': df['alert_aqi'].sum(),
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
names = list(alert_counts.keys()); vals = list(alert_counts.values())
colors_alert = [ORANGE, PURPLE, YELLOW, RED]
bars = ax1.barh(names, vals, color=colors_alert, alpha=0.85, edgecolor='none')
ax1.set_title('🚨  Alert Type Breakdown (7 Days)')
ax1.set_xlabel('Number of Hours')
ax1.grid(True, axis='x', alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
for bar, val in zip(bars, vals):
    ax1.text(val+0.3, bar.get_y()+bar.get_height()/2, str(int(val)), va='center', color='#E2E8F0', fontsize=10)

status_counts = {'Safe Air': (df['any_alert']==0).sum(), 'Alert Triggered': (df['any_alert']==1).sum()}
wedges, texts, autotexts = ax2.pie(
    status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%',
    colors=[GREEN, RED], startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0C0F14', linewidth=2),
    textprops={'color':'#E2E8F0','fontsize':10})
for at in autotexts: at.set_color('#0C0F14'); at.set_fontweight('bold')
ax2.set_title('🛡️  Air Quality Safety Overview')

plt.tight_layout()
plt.savefig('chart4_alerts.png', dpi=150, bbox_inches='tight', facecolor='#0C0F14')
plt.close(); print("Saved chart4")

# ── LIVE MONITORING SIMULATION ─────────────────────────────────
print("\n" + "="*78)
print("  LIVE AIR QUALITY MONITORING — LAST 10 READINGS")
print("="*78)
print(f"{'Time':<8} {'PM2.5':>7} {'AQI':>6} {'Category':>14} {'CO2':>6} {'CO':>6} {'Alert'}")
print("-"*78)
for _, row in df.tail(10).iterrows():
    alert_str = "⚠️ YES" if row['any_alert'] else "✅ OK"
    print(f"{row['timestamp'].strftime('%d %H:%M'):<8} {row['pm25']:>6.1f} {row['aqi']:>6.0f} "
          f"{row['aqi_category']:>14} {row['co2_ppm']:>6.0f} {row['co_ppm']:>6.2f}  {alert_str}")
print("="*78)

# ── FINAL REPORT ───────────────────────────────────────────────
safe_pct = (df['any_alert']==0).mean()*100
worst_hour = df.loc[df['aqi'].idxmax()]
print()
print("╔══════════════════════════════════════════════════════╗")
print("║   AIR QUALITY MONITORING DASHBOARD — REPORT         ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  📅 Monitoring Period : 7 Days (168 hourly readings)║")
print(f"║  🌫️  Avg PM2.5         : {df['pm25'].mean():.1f} µg/m³{'':<17}║")
print(f"║  📊 Avg AQI           : {df['aqi'].mean():.0f} ({aqi_category(df['aqi'].mean())}){'':<14}║")
print(f"║  🌡️  Avg Temperature   : {df['temperature_c'].mean():.1f}°C{'':<23}║")
print(f"║  💧 Avg Humidity      : {df['humidity_pct'].mean():.1f}%{'':<24}║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  🔴 Worst AQI         : {worst_hour['aqi']:.0f} ({worst_hour['aqi_category']}){'':<14}║")
print(f"║  🔴 Worst Time        : {worst_hour['timestamp'].strftime('%d %b %H:%M'):<28}║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  ⚠️  PM2.5 Alerts      : {int(df['alert_pm25'].sum()):<28}║")
print(f"║  ⚠️  CO₂ Alerts        : {int(df['alert_co2'].sum()):<28}║")
print(f"║  ⚠️  AQI Poor+ Alerts  : {int(df['alert_aqi'].sum()):<28}║")
print(f"║  ✅ Safe Air %        : {safe_pct:.1f}%{'':<24}║")
print("╠══════════════════════════════════════════════════════╣")
print("║  📁 Files Saved:                                     ║")
print("║     air_quality_data.csv                            ║")
print("║     chart1_air_quality_dashboard.png                ║")
print("║     chart2_aqi_breakdown.png                        ║")
print("║     chart3_aqi_heatmap.png                          ║")
print("║     chart4_alerts.png                               ║")
print("╚══════════════════════════════════════════════════════╝")
