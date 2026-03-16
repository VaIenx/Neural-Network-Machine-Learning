import matplotlib
matplotlib.use('Agg')  # Speichert Plots als Dateien – funktioniert immer in VS Code

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os


class Visualizer:
    """
    Debugging & Übersicht für die F1-Platzierings-Daten.
    Alle Plots helfen zu verstehen ob die Daten für das NN geeignet sind.

    Nutzung:
        viz = Visualizer(df)
        viz.plot_all()               # alle Plots auf einmal
        viz.gridpos_vs_position()    # einzelne Plots

    Speichern:
        viz = Visualizer(df, save_dir="plots")
        viz.plot_all()
    """

    TEAM_COLORS = {
        'Mercedes':        '#00D2BE',
        'Red Bull Racing': '#0600EF',
        'Ferrari':         '#DC0000',
        'McLaren':         '#FF8700',
        'Alpine':          '#0090FF',
        'AlphaTauri':      '#2B4562',
        'Aston Martin':    '#006F62',
        'Williams':        '#005AFF',
        'Alfa Romeo':      '#900000',
        'Haas F1 Team':    '#B0B0B0',
    }

    POS_COLORS = {
        'P1-3':   '#FFD700',
        'P4-6':   '#2ecc71',
        'P7-10':  '#3498db',
        'P11-15': '#e67e22',
        'P16-20': '#e74c3c',
        'DNF':    '#7f8c8d',
    }

    def __init__(self, dataframe: pd.DataFrame, save_dir: str = None):
        self.df = dataframe.copy()
        self.save_dir = save_dir

        self.df['Position'] = pd.to_numeric(self.df['Position'], errors='coerce')
        self.df['GridPosition'] = pd.to_numeric(self.df['GridPosition'], errors='coerce')
        self.df['pos_group'] = self.df['Position'].apply(self._pos_group)

        self._validate()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Plots werden gespeichert in: {os.path.abspath(self.save_dir)}/\n")

    # ─────────────────────────────────────────────
    #  HILFSMETHODEN
    # ─────────────────────────────────────────────

    def _pos_group(self, pos):
        if pd.isna(pos):
            return 'DNF'
        pos = int(pos)
        if pos <= 3:   return 'P1-3'
        if pos <= 6:   return 'P4-6'
        if pos <= 10:  return 'P7-10'
        if pos <= 15:  return 'P11-15'
        return 'P16-20'

    def _validate(self):
        required = ['Abbreviation', 'TeamName', 'GridPosition', 'Position']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

    def _save_or_show(self, filename: str):
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ok {filename}")
            plt.close()
        else:
            plt.show()

    def _team_color(self, team: str) -> str:
        return self.TEAM_COLORS.get(team, '#888888')

    def _pos_group_order(self):
        return [g for g in self.POS_COLORS if g in self.df['pos_group'].unique()]

    def _pos_legend_handles(self):
        return [
            mlines.Line2D([], [], color=self.POS_COLORS[g], marker='s',
                          linestyle='None', markersize=10, label=g)
            for g in self._pos_group_order()
        ]

    # ─────────────────────────────────────────────
    #  OEFFENTLICHE METHODEN
    # ─────────────────────────────────────────────

    def plot_all(self):
        print("Erstelle alle Plots...\n")
        self.gridpos_vs_position()
        self.avg_position_per_team()
        self.quali_vs_position()
        self.position_distribution()
        self.heatmap_team_gridpos()
        self.rainfall_effect()
        self.data_overview()
        print("\nFertig.")

    def gridpos_vs_position(self):
        """Startplatz vs. Endplatzierung – Kernplot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Startplatz vs. Endplatzierung", fontsize=14, fontweight='bold')

        df_clean = self.df.dropna(subset=['GridPosition', 'Position'])
        mean_pos = df_clean.groupby('GridPosition')['Position'].mean()
        std_pos  = df_clean.groupby('GridPosition')['Position'].std()

        jitter = np.random.uniform(-0.3, 0.3, len(df_clean))
        ax.scatter(df_clean['GridPosition'] + jitter,
                   df_clean['Position'],
                   alpha=0.08, s=12, color='steelblue', edgecolors='none',
                   label='Einzelergebnisse')

        ax.errorbar(mean_pos.index, mean_pos.values, yerr=std_pos.values,
                    fmt='o', color='#e74c3c', markersize=6, linewidth=1.5,
                    capsize=3, label='Mittelwert +/- Std')

        max_val = int(df_clean[['GridPosition', 'Position']].max().max()) + 1
        ax.plot([1, max_val], [1, max_val], '--', color='gray',
                linewidth=1.2, alpha=0.7, label='Ideallinie (Start = Ziel)')

        ax.set_xlabel("Startplatz (GridPosition)", fontsize=11)
        ax.set_ylabel("Endplatzierung (niedriger = besser)", fontsize=11)
        ax.set_xlim(0, max_val)
        ax.set_ylim(max_val + 1, 0)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        self._save_or_show("01_gridpos_vs_position.png")

    def avg_position_per_team(self):
        """Platzierungsverteilung pro Team (Boxplot)."""
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.suptitle("Platzierungsverteilung pro Team", fontsize=14, fontweight='bold')

        df_clean = self.df.dropna(subset=['Position'])
        teams = (df_clean.groupby('TeamName')['Position']
                 .mean().sort_values().index.tolist())

        data_per_team = [df_clean[df_clean['TeamName'] == t]['Position'].values for t in teams]
        colors = [self._team_color(t) for t in teams]

        bp = ax.boxplot(data_per_team, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        for i, data in enumerate(data_per_team):
            ax.plot(i + 1, np.mean(data), 'D', color='black', markersize=5, zorder=5)

        ax.set_xticks(range(1, len(teams) + 1))
        ax.set_xticklabels(teams, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel("Endplatzierung (niedriger = besser)", fontsize=11)
        ax.set_xlabel("Team", fontsize=11)
        ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
        ax.grid(True, alpha=0.25, axis='y')

        team_handles = [
            mlines.Line2D([], [], color=self._team_color(t), marker='s',
                          linestyle='None', markersize=9, label=t)
            for t in teams
        ]
        mean_h   = mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=6, label='Mittelwert')
        median_h = mlines.Line2D([], [], color='black', linewidth=2, label='Median')
        ax.legend(handles=team_handles + [mean_h, median_h], fontsize=8, loc='upper right', ncol=2)

        plt.tight_layout()
        self._save_or_show("02_position_per_team.png")

    def quali_vs_position(self):
        """Qualizeit vs. Endplatzierung, eingefaerbt nach Positionsgruppe."""
        if 'Q_best_sec' not in self.df.columns:
            print("  - Q_best_sec nicht vorhanden, Plot uebersprungen.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Qualizeit vs. Endplatzierung", fontsize=14, fontweight='bold')

        df_clean = self.df.dropna(subset=['Q_best_sec', 'Position'])

        for group in self._pos_group_order():
            sub = df_clean[df_clean['pos_group'] == group]
            ax.scatter(sub['Q_best_sec'], sub['Position'],
                       color=self.POS_COLORS[group], alpha=0.45, s=18,
                       edgecolors='none', label=group)

        bins = pd.cut(df_clean['Q_best_sec'], bins=25)
        trend = df_clean.groupby(bins, observed=True)['Position'].median()
        centers = [b.mid for b in trend.index]
        ax.plot(centers, trend.values, color='black', linewidth=2,
                linestyle='--', label='Median-Trend')

        ax.set_xlabel("Beste Qualizeit Q_best_sec (Sekunden)", fontsize=11)
        ax.set_ylabel("Endplatzierung (niedriger = besser)", fontsize=11)
        ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
        ax.legend(fontsize=9, title="Positionsgruppe")
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        self._save_or_show("03_quali_vs_position.png")

    def position_distribution(self):
        """Verteilung der Endplatzierungen – gesamt und pro Jahr."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Verteilung der Endplatzierungen", fontsize=14, fontweight='bold')

        df_clean = self.df.dropna(subset=['Position'])
        pos_counts = df_clean['Position'].value_counts().sort_index()
        bar_colors = [self.POS_COLORS[self._pos_group(p)] for p in pos_counts.index]

        axes[0].bar(pos_counts.index, pos_counts.values,
                    color=bar_colors, edgecolor='black', linewidth=0.4)
        axes[0].set_xlabel("Endplatzierung", fontsize=10)
        axes[0].set_ylabel("Anzahl Fahrer", fontsize=10)
        axes[0].set_title("Haeufigkeit jeder Platzierung")
        axes[0].grid(True, alpha=0.25, axis='y')
        axes[0].legend(handles=self._pos_legend_handles(), title="Positionsgruppe", fontsize=8)

        if 'year' in self.df.columns:
            pivot = (self.df.groupby(['year', 'pos_group'])
                     .size().unstack(fill_value=0))
            group_order = [g for g in self.POS_COLORS if g in pivot.columns]
            pivot = pivot[group_order]
            colors = [self.POS_COLORS[g] for g in group_order]
            pivot.plot(kind='bar', ax=axes[1], color=colors,
                       edgecolor='black', linewidth=0.4, width=0.75)
            axes[1].set_title("Positionsgruppen pro Jahr")
            axes[1].set_xlabel("Jahr", fontsize=10)
            axes[1].set_ylabel("Anzahl Fahrer", fontsize=10)
            axes[1].tick_params(axis='x', rotation=0)
            axes[1].legend(title="Positionsgruppe", fontsize=8)
            axes[1].grid(True, alpha=0.25, axis='y')
        else:
            grp_counts = self.df['pos_group'].value_counts()
            grp_order  = [g for g in self.POS_COLORS if g in grp_counts.index]
            axes[1].pie(
                [grp_counts[g] for g in grp_order],
                labels=grp_order,
                colors=[self.POS_COLORS[g] for g in grp_order],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}
            )
            axes[1].set_title("Positionsgruppen gesamt")

        plt.tight_layout()
        self._save_or_show("04_position_distribution.png")

    def heatmap_team_gridpos(self):
        """Heatmap: Startplatz x Team -> mittlere Endplatzierung."""
        fig, ax = plt.subplots(figsize=(15, 7))
        fig.suptitle("Mittlere Endplatzierung: Startplatz x Team",
                     fontsize=14, fontweight='bold')

        df_clean = self.df.dropna(subset=['GridPosition', 'Position'])
        top_grid = sorted(df_clean['GridPosition'].unique())[:21]
        df_f = df_clean[df_clean['GridPosition'].isin(top_grid)]

        pivot = df_f.pivot_table(
            values='Position', index='TeamName',
            columns='GridPosition', aggfunc='mean'
        )

        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=20)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Durchschnittliche Endplatzierung (niedriger = besser)', fontsize=10)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(c) for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Startplatz", fontsize=11)
        ax.set_ylabel("Team", fontsize=11)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    text_color = 'white' if (val < 5 or val > 16) else 'black'
                    ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                            fontsize=7, color=text_color)

        plt.tight_layout()
        self._save_or_show("05_heatmap_team_gridpos.png")

    def rainfall_effect(self):
        """Regen vs. Trocken: Auswirkung auf Endplatzierung."""
        if 'rainfall' not in self.df.columns:
            print("  - rainfall nicht vorhanden, Plot uebersprungen.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Regen-Effekt auf Endplatzierung", fontsize=14, fontweight='bold')

        df_clean = self.df.dropna(subset=['GridPosition', 'Position'])
        styles = [
            (False, 'Trocken', '#3498db'),
            (True,  'Regen',   '#95a5a6'),
        ]

        for rain, label, color in styles:
            sub = df_clean[df_clean['rainfall'] == rain]
            mean_pos = sub.groupby('GridPosition')['Position'].mean()
            axes[0].plot(mean_pos.index, mean_pos.values,
                         label=f"{label} (n={len(sub)})",
                         color=color, linewidth=2, marker='o', markersize=4)

        axes[0].plot([1, 20], [1, 20], '--', color='gray',
                     linewidth=1, alpha=0.6, label='Ideallinie')
        axes[0].set_xlabel("Startplatz", fontsize=10)
        axes[0].set_ylabel("Durchschnittliche Endplatzierung", fontsize=10)
        axes[0].set_title("Startplatz -> Durchschnittliche Endplatzierung")
        axes[0].set_ylim(axes[0].get_ylim()[1], axes[0].get_ylim()[0])
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.25)

        grp_order = [g for g in self.POS_COLORS if g in self.df['pos_group'].unique()]
        x = np.arange(len(grp_order))
        width = 0.35

        for idx, (rain, label, color) in enumerate(styles):
            sub = self.df[self.df['rainfall'] == rain]
            counts = sub['pos_group'].value_counts()
            total  = len(sub)
            vals   = [(counts.get(g, 0) / total * 100) for g in grp_order]
            offset = (idx - 0.5) * width
            axes[1].bar(x + offset, vals, width,
                        label=label, color=color,
                        edgecolor='black', linewidth=0.5, alpha=0.85)

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(grp_order, fontsize=9)
        axes[1].set_ylabel("Anteil Fahrer (%)", fontsize=10)
        axes[1].set_xlabel("Positionsgruppe", fontsize=10)
        axes[1].set_title("Positionsgruppen: Trocken vs. Regen")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.25, axis='y')

        plt.tight_layout()
        self._save_or_show("06_rainfall_effect.png")

    def data_overview(self):
        """Uebersichts-Dashboard: Verteilungen + Kennzahlen-Tabelle."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Daten-Uebersicht", fontsize=14, fontweight='bold')

        df_clean = self.df.dropna(subset=['GridPosition', 'Position'])

        # Oben links: GridPosition Histogramm
        axes[0, 0].hist(df_clean['GridPosition'], bins=20,
                        color='steelblue', edgecolor='black', linewidth=0.4, alpha=0.85)
        axes[0, 0].set_title("Startplatz-Verteilung")
        axes[0, 0].set_xlabel("Startplatz")
        axes[0, 0].set_ylabel("Anzahl Fahrer")
        axes[0, 0].grid(True, alpha=0.25)
        axes[0, 0].legend(handles=[
            mlines.Line2D([], [], color='steelblue', marker='s',
                          linestyle='None', markersize=9, label='Anzahl Starts')
        ], fontsize=8)

        # Oben rechts: Q_best_sec Histogramm
        if 'Q_best_sec' in self.df.columns:
            q_clean = self.df['Q_best_sec'].dropna()
            axes[0, 1].hist(q_clean, bins=40,
                            color='darkorange', edgecolor='black', linewidth=0.4, alpha=0.85)
            axes[0, 1].axvline(q_clean.mean(), color='red', linestyle='--',
                               linewidth=1.5, label=f"Mittelwert: {q_clean.mean():.2f}s")
            axes[0, 1].axvline(q_clean.median(), color='navy', linestyle=':',
                               linewidth=1.5, label=f"Median: {q_clean.median():.2f}s")
            axes[0, 1].set_title("Qualizeit-Verteilung (Q_best_sec)")
            axes[0, 1].set_xlabel("Qualizeit (Sekunden)")
            axes[0, 1].set_ylabel("Anzahl Fahrer")
            axes[0, 1].legend(fontsize=8)
        else:
            axes[0, 1].text(0.5, 0.5, 'Q_best_sec\nnicht vorhanden',
                            ha='center', va='center', transform=axes[0, 1].transAxes,
                            fontsize=12, color='gray')
        axes[0, 1].grid(True, alpha=0.25)

        # Unten links: Datenmenge pro Jahr
        if 'year' in self.df.columns:
            year_counts = self.df['year'].value_counts().sort_index()
            bars = axes[1, 0].bar(year_counts.index.astype(str), year_counts.values,
                                  color='mediumseagreen', edgecolor='black', linewidth=0.5)
            for bar, cnt in zip(bars, year_counts.values):
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 1, str(cnt),
                                ha='center', fontsize=9, fontweight='bold')
            axes[1, 0].set_title("Datenmenge pro Jahr")
            axes[1, 0].set_xlabel("Jahr")
            axes[1, 0].set_ylabel("Anzahl Einträge")
            axes[1, 0].legend(handles=[
                mlines.Line2D([], [], color='mediumseagreen', marker='s',
                              linestyle='None', markersize=9, label='Eintraege pro Jahr')
            ], fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'Keine year-Spalte',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].grid(True, alpha=0.25, axis='y')

        # Unten rechts: Kennzahlen-Tabelle
        axes[1, 1].axis('off')
        total = len(self.df)
        rain_races = self.df[self.df['rainfall'] == True]['race'].nunique() if 'rainfall' in self.df.columns else '-'
        stats = [
            ["Gesamt Eintraege",        str(total)],
            ["Rennen",                  str(self.df['race'].nunique()) if 'race' in self.df.columns else '-'],
            ["Teams",                   str(self.df['TeamName'].nunique())],
            ["Fahrer",                  str(self.df['Abbreviation'].nunique())],
            ["Jahre",                   str(sorted(self.df['year'].unique().tolist())) if 'year' in self.df.columns else '-'],
            ["Durchschn. Endplatzierung", f"{self.df['Position'].mean():.1f}"],
            ["Median Endplatzierung",   f"{self.df['Position'].median():.1f}"],
            ["Regenrennen",             str(rain_races)],
            ["Fehlende Werte",          str(int(self.df.isnull().sum().sum()))],
        ]
        table = axes[1, 1].table(
            cellText=stats,
            colLabels=["Kennzahl", "Wert"],
            loc='center', cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.7)
        axes[1, 1].set_title("Zusammenfassung", fontweight='bold', pad=12)

        plt.tight_layout()
        self._save_or_show("07_data_overview.png")
    
    def plot_training(self, loss_list, val_loss_list):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Training Übersicht", fontsize=14, fontweight='bold')

        ax.plot(loss_list, color='steelblue', linewidth=2, label='Train Loss')
        ax.plot(val_loss_list, color='darkorange', linewidth=2, label='Validation Loss')
        ax.set_xlabel("Epoche")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        self._save_or_show("08_training_overview.png")