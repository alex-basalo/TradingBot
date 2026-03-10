# **Hybrider Clustering-Ansatz aus Literaturrecherche**

Dieser Ansatz kombiniert die Extraktion von aussagekräftigen Finanzkennzahlen mit einem zeitreihenspezifischen Gruppierungsverfahren, um Finanzwerte (Assets) mit ähnlichem Marktverhalten in Cluster zusammenzufassen. Der Prozess gliedert sich in vier wesentliche Schritte:

1. *Datenaufbereitung und Feature-Extraktion*:
Anstatt lediglich die reinen, absoluten Preisverläufe der Assets zu betrachten, werden die Daten zunächst in normalisierte Zeitreihen und relevante Finanzmetriken (Features) transformiert. Hierbei werden für jedes Asset Eigenschaften wie tägliche Renditen, die rollierende Volatilität (Risiko), Drawdowns oder Momentum-Indikatoren berechnet. Dies stellt sicher, dass Assets nicht aufgrund ihrer unterschiedlichen Preisniveaus (z. B. ein Asset bei 60.000 € vs. eines bei 0,50 €), sondern aufgrund ihrer tatsächlichen Marktdynamik objektiv verglichen werden.

2. *Distanzmessung mittels Dynamic Time Warping (DTW)*:
Um zu berechnen, wie ähnlich sich die Verhaltensmuster zweier Assets sind, wird das Distanzmaß Dynamic Time Warping (DTW) verwendet.
Im Gegensatz zu klassischen Methoden, die Datenpunkte starr zum exakt selben Zeitpunkt vergleichen, kann DTW zeitliche Verzögerungen (Phasenverschiebungen) abgleichen. Reagiert beispielsweise ein Asset sofort auf ein Marktereignis und ein zweites Asset zeigt exakt dasselbe Kursmuster, jedoch erst mit einem Tag Verzögerung, erkennt DTW diese strukturelle Ähnlichkeit.

3. *Gruppierung durch K-Means Clustering*:
Basierend auf den mit DTW berechneten Distanzen wird der K-Means-Algorithmus angewendet.
Dieser Algorithmus teilt die Menge der Assets iterativ in eine bestimmte Anzahl von Gruppen (Clustern) ein. Das Ziel des Algorithmus ist es, die Zuweisungen so zu optimieren, dass die Assets innerhalb eines Clusters extrem ähnliche Verhaltensmuster aufweisen, während sich die Cluster untereinander so stark wie möglich unterscheiden. In Python lässt sich dies effizient über die Bibliothek tslearn umsetzen, welche K-Means direkt mit DTW verknüpft.

4. *Evaluierung und Bestimmung der Cluster-Anzahl*:
Da die optimale Anzahl der Cluster (k) für die 100 Assets im Voraus unbekannt ist, wird das Clustering-Verfahren für verschiedene Werte (z. B. k=3 bis k=15) mehrfach durchgeführt. Die Qualität der jeweiligen Gruppierungen wird anschließend durch den sogenannten Silhouette Score bewertet. Diese Metrik misst, wie gut jedes Asset zu seinem zugewiesenen Cluster im Vergleich zu den benachbarten Clustern passt. Der Durchlauf, der den höchsten Silhouette Score erzielt, definiert die mathematisch fundierteste und endgültige Cluster-Anzahl für die Konfiguration des Trading-Bots.


---

**Quellen**

Nachfolgend die Quellen, welche nach breiter Recherche konkret als Lösungsansatz dienen und an denen der oben aufgeführte Plan orientiert ist

*Für 1.*

R. Setiawan and M. S. Hakim, "Diversified Crypto Assets Portfolio Optimization Using K-Means Clustering Algorithm And The Efficient Frontier," 2023 IEEE Technology & Engineering Management Conference - Asia Pacific (TEMSCON-ASPAC), Bengaluru, India, 2023, pp. 1-6, doi: 10.1109/TEMSCON-ASPAC59527.2023.10531468.

Y. Zhang, H. Zhao, X. Li, S. Gao, K. Kong and Y. Chen, "Exchange Traded Fund Clustering via Metric Learning," 2020 IEEE International Conference on Big Data (Big Data), Atlanta, GA, USA, 2020, pp. 5486-5495, doi: 10.1109/BigData50022.2020.9378205.

*Für 2.*

H. W. Aqsari, D. D. Prastyo and S. Puteri Rahayu, "Clustering Stock Prices of Financial Sector Using K-Means Clustering With Dynamic Time Warping," 2022 6th International Conference on Information Technology, Information Systems and Electrical Engineering (ICITISEE), Yogyakarta, Indonesia, 2022, pp. 503-507, doi: 10.1109/ICITISEE57756.2022.10057714.

*Für 3.*

R. Setiawan and M. S. Hakim, "Diversified Crypto Assets Portfolio Optimization Using K-Means Clustering Algorithm And The Efficient Frontier," 2023 IEEE Technology & Engineering Management Conference - Asia Pacific (TEMSCON-ASPAC), Bengaluru, India, 2023, pp. 1-6, doi: 10.1109/TEMSCON-ASPAC59527.2023.10531468.

H. W. Aqsari, D. D. Prastyo and S. Puteri Rahayu, "Clustering Stock Prices of Financial Sector Using K-Means Clustering With Dynamic Time Warping," 2022 6th International Conference on Information Technology, Information Systems and Electrical Engineering (ICITISEE), Yogyakarta, Indonesia, 2022, pp. 503-507, doi: 10.1109/ICITISEE57756.2022.10057714.

*Ergänzend*

Y. -C. Hsu and A. -P. Chen, "Clustering Time Series Data by SOM for the Optimal Hedge Ratio Estimation," 2008 Third International Conference on Convergence and Hybrid Information Technology, Busan, Korea (South), 2008, pp. 1164-1169, doi: 10.1109/ICCIT.2008.408.

X. Huang, "Research on Financial Time Series Risk Assessment Model Based on Computer Deep Learning," 2024 2nd International Conference on Mechatronics, IoT and Industrial Informatics (ICMIII), Melbourne, Australia, 2024, pp. 450-453, doi: 10.1109/ICMIII62623.2024.00088.

H. Ziegler, M. Jenny, T. Gruse and D. A. Keim, "Visual market sector analysis for financial time series data," 2010 IEEE Symposium on Visual Analytics Science and Technology, Salt Lake City, UT, USA, 2010, pp. 83-90, doi: 10.1109/VAST.2010.5652530.

S. Begušić and Z. Kostanjčar, "Cluster-Specific Latent Factor Estimation in High-Dimensional Financial Time Series," in IEEE Access, vol. 8, pp. 164365-164379, 2020, doi: 10.1109/ACCESS.2020.3021898.

---

| Quelle | Inhaltlicher Fokus & Methode | Bezug zum eigenen Vorhaben | Status im Projekt |
| :--- | :--- | :--- | :--- |
| **Setiawan & Hakim (2023):** *Diversified Crypto Assets Portfolio Optimization Using K-Means* | **Portfolio-Optimierung:** K-Means Clustering zur Konstruktion diversifizierter Portfolios. | Beweist empirisch die Eignung von K-Means zur Gruppierung von Assets und zur Filterung von Rauschen. | **Eingeflossen** (Schritt 1 & 3) |
| **Aqsari et al. (2022):** *Clustering Stock Prices Using K-Means Clustering With DTW* | **Zeitreihen-Metriken:** Nutzung von Dynamic Time Warping (DTW) mit K-Means zur Aktien-Gruppierung. | Liefert die methodische Rechtfertigung für die Wahl von DTW, um zeitliche Phasenverschiebungen abzubilden. | **Eingeflossen** (Schritt 2 & 3) |
| **Zhang et al. (2020):** *Exchange Traded Fund Clustering via Metric Learning* | **Portfolio-Allokation:** Metric Learning (ITML) kombiniert mit Finanzmetriken. | Begründet den Ansatz, nicht nur absolute Preise, sondern normalisierte finanzielle Features zu clustern. | **Eingeflossen** (Schritt 1) |
| **Gu, Kelly, & Xiu (2020):** *Empirical Asset Pricing via Machine Learning* | **Asset Pricing:** Empirische Untersuchung von Machine-Learning-Verfahren zur Renditevorhersage. | Dient als wissenschaftliches Standardwerk zur Validierung der extrahierten Finanz-Features. | **Ergänzend** |
| **Asness et al. (2015):** *Investing with Style* | **Faktor-Modelle:** "Style Investing" (z. B. Momentum, Value) über verschiedene Assetklassen hinweg. | Liefert die akademische Basis für Cross-Asset-Features, die in das Clustering einfließen. | **Ergänzend** |
| **Bali et al. (2021):** *Different Strokes: Return Predictability across Stocks and Bonds* | **Cross-Asset ML:** Renditevorhersage über Aktien und Anleihen mittels Machine Learning. | Untermauert die Übertragbarkeit von ML-Verfahren auf ein heterogenes Portfolio (100 Assets). | **Ergänzend** |
| **Begušić & Kostanjčar (2020):** *Cluster-Specific Latent Factor Estimation* | **Faktor-Modellierung:** Spektrales Clustering zur Identifikation latenter Faktoren. | Bietet das mathematische Fundament zur Trennung von globalen Markttrends und spezifischen Asset-Bewegungen. | **Ergänzend** |
| **Huang (2024):** *Financial Time Series Risk Assessment Model Based on Deep Learning* | **Anomalie-Erkennung:** Nutzung von GCN-LSTM-Clustering zur Risikoidentifikation in Zeitreihen. | Bietet theoretischen Hintergrund für die Identifikation von Ausreißern und Marktanomalien. | **Ergänzend** |
| **Hsu & Chen (2008):** *Clustering Time Series Data by SOM* | **Pattern Recognition:** Self-Organizing Maps (SOM) für Finanzdaten mit "Fat Tails". | Dient als methodische Alternative zur Erkennung topologischer Muster im Marktverhalten. | **Ergänzend** |
| **Ziegler et al. (2010):** *Visual Market Sector Analysis for Financial Time Series Data* | **Explorative Datenanalyse:** Interaktive Visualisierung von Zeitreihen und Sektor-Clustern. | Betont die Wichtigkeit der visuellen Überwachung von Clustern und Marktturbulenzen. | **Ergänzend** |
| **Puspita et al. (2020):** *A Practical Evaluation of Dynamic Time Warping in Financial Time Series Clustering* | **Zeitreihen-Metriken:** Vergleich von DTW und Euklidischer Distanz unter Berücksichtigung der Rechenzeit. | Veranschaulicht den methodischen Trade-off zwischen Recheneffizienz und Datenhomogenität beim Clustern. | **Alternative** |
| **Xiao et al. (2013):** *The Research of Morphological Characteristics in Time Series of Stock Prices Based on CBR* | **Case-Based Reasoning:** Mustererkennung (Turning Points) basierend auf historischen Fällen. | Stellt einen alternativen Ansatz zur konkreten Signalgenerierung innerhalb bestehender Cluster dar. | **Alternative** |
| **Leonarduzzi et al. (2019):** *Maximum-entropy Scattering Models for Financial Time Series* | **Signalverarbeitung:** Untersuchung von Heavy Tails mittels Wavelet-Scattering. | Hochkomplexer Ansatz für die mathematische Modellierung von Signalrauschen; für den aktuellen Scope zu rechenintensiv. | **Alternative** |
| **Gottimukkala (2024):** *Applying the Multifractal Model of Asset Returns (MMAR) to Financial Markets* | **Generative Modelle:** Fraktale Simulationen von Finanzmärkten. | Eignet sich primär für Portfolio-Stresstests (Crash-Simulationen), nicht für die Live-Gruppierung. | **Alternative** |
| **Silva et al. (2017):** *Interval Fuzzy Rule-based Modeling Approach for Financial Time Series* | **Intervall-Vorhersage:** Fuzzy-Logik für High/Low-Zeitreihen. | Bietet ein alternatives Modell für dynamisches Risikomanagement anstelle von Punkt-Schätzern. | **Alternative** |
| **Silva et al. (2015):** *Evolving Possibilistic Fuzzy Modeling for Financial Interval Time Series* | **Echtzeit-Adaption:** Rekursives Clustering von Stream-Daten. | Stellt eine Methode für Live-Updates dar, welche bei 100 Assets jedoch zu Performance-Problemen führen könnte. | **Alternative** |
