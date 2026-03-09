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
