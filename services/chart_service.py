"""
Chart generation service with Sephira branding.
Creates time series charts with watermarks, footers, and legal disclaimers.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64
import io
import logging

from config import Config

logger = logging.getLogger(__name__)


class ChartService:
    """Service for generating branded charts."""
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        """
        Initialize chart service.
        
        Args:
            output_dir: Directory to save generated charts
            dpi: Dots per inch for chart output
        """
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('dark_background')
    
    def generate_chart(self, data: Dict[str, Any], countries: List[str],
                      date_range: Dict[str, str], chart_type: str,
                      title: str) -> Dict[str, str]:
        """
        Generate a chart with Sephira branding.
        
        Args:
            data: DataFrame with date and country columns
            countries: List of country names to plot
            date_range: Dict with 'start' and 'end' dates
            chart_type: Type of chart ('time_series', 'comparison', 'regional')
            title: Chart title
        
        Returns:
            Dict with 'chart_url' and 'base64_image'
        """
        try:
            # Create figure with Sephira styling
            fig = self._create_figure()
            
            # Create main plot area
            ax = self._create_plot_area(fig)
            
            # Plot data based on chart type
            if chart_type == "time_series":
                self._plot_time_series(ax, data, countries, date_range)
            elif chart_type == "comparison":
                self._plot_comparison(ax, data, countries, date_range)
            else:
                self._plot_time_series(ax, data, countries, date_range)  # Default
            
            # Add title
            ax.set_title(title, color=Config.COLOR_TEXT_PRIMARY, 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Add Sephira branding elements
            self._add_watermark(fig)
            self._add_footer(fig, date_range)
            
            # Save to file and generate base64
            chart_filename = self._generate_filename(countries, date_range)
            chart_path = self.output_dir / chart_filename
            
            fig.savefig(chart_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=Config.COLOR_BG_PRIMARY,
                       edgecolor='none')
            
            # Generate base64 for API response
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor=Config.COLOR_BG_PRIMARY,
                       edgecolor='none')
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
            logger.info(f"Generated chart: {chart_filename}")
            
            return {
                "chart_url": f"/static/charts/{chart_filename}",
                "base64_image": base64_image,
                "filename": chart_filename
            }
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            plt.close('all')  # Close any open figures
            raise
    
    def _create_figure(self) -> plt.Figure:
        """Create figure with Sephira background."""
        fig = plt.figure(figsize=(14, 8), facecolor=Config.COLOR_BG_PRIMARY)
        return fig
    
    def _create_plot_area(self, fig: plt.Figure) -> plt.Axes:
        """Create plot area with gradient background."""
        # Create gradient background for plot area
        ax = fig.add_subplot(111, facecolor=Config.COLOR_BG_SECONDARY)
        
        # Create gradient effect using fill_between (simplified approach)
        # In a full implementation, you'd use a more sophisticated gradient
        
        # Set plot styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(Config.COLOR_TEXT_SECONDARY)
        ax.spines['left'].set_color(Config.COLOR_TEXT_SECONDARY)
        
        ax.tick_params(colors=Config.COLOR_TEXT_SECONDARY)
        ax.xaxis.label.set_color(Config.COLOR_TEXT_PRIMARY)
        ax.yaxis.label.set_color(Config.COLOR_TEXT_PRIMARY)
        
        ax.grid(True, alpha=0.2, color=Config.COLOR_TEXT_SECONDARY)
        
        # Add rounded corners effect (visual approximation)
        ax.set_xlabel('Date', color=Config.COLOR_TEXT_PRIMARY, fontsize=12)
        ax.set_ylabel('Sentiment Index', color=Config.COLOR_TEXT_PRIMARY, fontsize=12)
        
        return ax
    
    def _plot_time_series(self, ax: plt.Axes, data: Any, countries: List[str],
                         date_range: Dict[str, str]):
        """Plot time series data for countries."""
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            if 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
                
                # Plot each country
                colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
                
                for i, country in enumerate(countries):
                    if country in data.columns:
                        values = data[country].dropna()
                        valid_dates = dates[data[country].notna()]
                        
                        if len(values) > 0:
                            ax.plot(valid_dates, values, 
                                  label=country, linewidth=2.5,
                                  color=colors[i], alpha=0.9)
                
                # Format x-axis dates
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add legend
                if len(countries) > 0:
                    legend = ax.legend(loc='upper left', frameon=True,
                                     facecolor=Config.COLOR_BG_SECONDARY,
                                     edgecolor=Config.COLOR_TEXT_SECONDARY,
                                     labelcolor=Config.COLOR_TEXT_PRIMARY)
                    legend.get_frame().set_alpha(0.8)
    
    def _plot_comparison(self, ax: plt.Axes, data: Any, countries: List[str],
                        date_range: Dict[str, str]):
        """Plot comparison chart (can be enhanced for side-by-side)."""
        # For now, use time series approach
        self._plot_time_series(ax, data, countries, date_range)
    
    def _add_watermark(self, fig: plt.Figure):
        """Add Sephira copyright watermark."""
        watermark_text = "© Sephira"
        
        fig.text(0.98, 0.02, watermark_text,
                fontsize=10, color=Config.COLOR_TEXT_SECONDARY,
                ha='right', va='bottom', alpha=0.5,
                fontweight='bold')
    
    def _add_footer(self, fig: plt.Figure, date_range: Dict[str, str]):
        """Add footer with data source attribution and legal disclaimer."""
        current_year = datetime.now().year
        
        # Data source attribution
        data_source = "Data Source: Sephira Sentiment Index"
        
        # Legal disclaimer
        disclaimer = "© Sephira {} | This data is proprietary and confidential. Unauthorized reproduction prohibited.".format(current_year)
        
        # Combine footer text
        footer_text = f"{data_source} | {disclaimer}"
        
        # Add footer at bottom
        fig.text(0.5, 0.01, footer_text,
                fontsize=8, color=Config.COLOR_TEXT_SECONDARY,
                ha='center', va='bottom', alpha=0.7,
                wrap=True)
    
    def _generate_filename(self, countries: List[str], date_range: Dict[str, str]) -> str:
        """Generate unique filename for chart."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country_str = "_".join(countries[:3])  # Max 3 countries in filename
        if len(countries) > 3:
            country_str += f"_and_{len(countries)-3}more"
        
        start_date = date_range.get('start', '')[:10].replace('-', '')
        end_date = date_range.get('end', '')[:10].replace('-', '')
        
        filename = f"chart_{country_str}_{start_date}_{end_date}_{timestamp}.png"
        
        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        return filename

