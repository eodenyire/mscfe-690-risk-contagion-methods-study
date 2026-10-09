# Literature Review and Competitor Analysis

## Literature Review: Financial Contagion and Risk Interconnectedness

There is a growing body of literature that studies financial interconnectedness, financial contagion, and bank failures. The overwhelming evidence from the literature suggests that bank failures rarely result from a single isolated risk failure. They usually arise when the failure in one Primary Risk Type (PRT) spreads to other PRTs within an institution, resulting in distress transmitted either within an institution (intra-institutional) or across institutions (inter-institutional) due to interconnectedness within the wider system. The literature has traditionally addressed intra- and inter-institutional risk interrelatedness, risk failure, and spread largely separately, with each believed to ultimately result in contagion and systemic failure.

### Intra-Firm Risk Interconnectedness

The intra-firm interrelatedness tranche of the literature can be traced to Diamond and Dybvig (1983), whose seminal paper "Bank Runs, Deposit Insurance and Liquidity" documented the first formal account of how the failure of one risk type—bank runs (liquidity risk)—forces a solvent bank to liquidate illiquid assets mostly at a loss, thus turning liquidity risk into solvency risk. Specifically, Diamond and Dybvig (1983) established a theoretical framework studying how liquidity crises propagate through the banking system. Their model demonstrated that the nature of bank deposit contracts ensures banks are inherently vulnerable to self-fulfilling runs due to maturity mismatch between illiquid assets and liquid liabilities. Bank runs can eventually result in solvency risk for an institution if significant illiquid asset liquidation is required to fulfill customer withdrawal requests.

He and Xiong (2012) show that creditors' fear of rollover risks (liquidity risk) can raise the default probability of even solvent banks. They argue that banks have multiple creditors, and each creditor deciding whether to roll over maturing debt considers the rollover decisions of other creditors whose debt matures during the next period. A maturing creditor will choose to roll over debt if and only if current fundamentals provide sufficient safety margin.

Brunnermeier and Pedersen (2009) theoretically document that market and funding liquidity reinforce each other in a spiral where falling asset prices tighten margins and funding, forcing the firm into further sales. This feedback loop exemplifies how one risk type (market liquidity) can amplify another (funding liquidity), ultimately affecting solvency.

### Inter-Institutional Risk Contagion

The inter-institutional tranche of the literature traces to Allen and Gale (2000) and Freixas, Parigi, and Rocher (2000). These papers are credited for formalizing contagion as the transmission of distress through overlapping interbank claims across regions or sectors of the banking system. Allen and Gale (2000) demonstrated that the structure of the interbank network—whether complete or incomplete—fundamentally determines how shocks propagate. In complete interbank networks where all banks have claims on each other and each region is connected to all others, contagion is limited because losses are diversified. In incomplete interbank networks where each bank/region is connected to a small number of other banks/regions, the failure of a single bank can trigger a cascade of insolvencies across neighboring regions.

Eisenberg and Noe (2001) provided the closest demonstration of inter-institutional contagion effects of risk failures. In "Systemic Risk in Financial Systems," they theoretically document inter-institutional risk contagion by developing a clearing payment vector model demonstrating how a single firm's default can trigger insolvency down a chain of balance sheet liabilities. By modeling financial institutions as a network of mutual obligations where one institution's ability to meet its liabilities depends partly on payments received from others, Eisenberg and Noe (2001) showed that interdependence can amplify individual shocks or risk failures into systemic risk failures. Their model formalized the concept of cascading failures in financial systems and provided foundations for subsequent research on financial contagion.

### Empirical Evidence of Risk Interconnectedness

Empirically, the intra- and inter-institutional contagion effects of risk failures have been documented by several papers. Imbierowicz and Rauch (2014) empirically document the liquidity-credit risk linkage for 15,000 US banks and find that each risk type raises bank default probabilities, and their interaction significantly adds to default probabilities. Acharya, Pedersen, Philippon, and Richardson (2017) demonstrated that tail dependencies between institutions are critical for understanding systemic risk. In response, Acharya et al. (2017) developed Systemic Expected Shortfall (SES) as a measure capturing the expected capital shortfall of a financial institution conditional on the system being in distress.

Other empirical papers document indirect channels through which financial distress spreads. Greenwood, Landier, and Thesmar (2015) use European bank balance sheet data to quantify spillovers from fire-sale dynamics, arguing that common asset exposures can amplify losses when banks deleverage simultaneously. Acharya and Skeie (2011) document a liquidity channel where heightened rollover risk induces banks to hoard liquidity and reduce interbank lending, contributing to higher funding costs for other firms. The 2007-2008 financial crisis presented perhaps the best anecdotal evidence of interlinkages between firms and financial contagion effects. Gorton and Metrick (2012), using data on securitized bond spreads and repo-market conditions, find evidence that the crisis propagated through the repo market with rising counterparty risk and falling collateral values contributing to overall financial crisis.

### African Banking Context

The experience within the African continent parallels global patterns. Bello, Guo, and Newaz (2022) using dynamic correlations for African markets between 2005-2020, focusing on the European debt crisis, Brexit, and COVID-19, find that contagion exists for some markets with regional evidence of significant contagion strongest during the global financial crisis. They also find that contagion varies with country-level risk, market capitalization, and export exposure. Boako and Alagidede (2018) find that African equity-market correlation increased significantly during the global financial crisis.

The challenge for African financial institutions is compounded by data constraints. Confidentiality and regulatory privacy requirements prevent institutions from sharing granular internal risk data for research purposes. This data scarcity limits the ability to study actual risk interconnectedness patterns within African banking systems, making methodological frameworks for controlled benchmarking of recovery techniques particularly valuable.

## Methodological Gap: Connecting Literature to Methods

While the existing literature extensively documents financial contagion and network spillovers, a significant methodological gap remains in identifying how to benchmark network-recovery techniques under data-scarce conditions. The circularity problem arises when one generates synthetic data with an embedded dependency structure and then uses correlations, Granger causality, or machine learning to "discover" that same structure—recovering only what was artificially hardcoded.

Our study directly bridges theoretical models with practical recovery techniques by establishing a rigorous methods-study framework. We utilize:

1. **Theoretical grounding** from Eisenberg and Noe (2001) and the empirical connectedness methods of Diebold and Yilmaz (2014) and Adrian and Brunnermeier (2016).

2. **Tail dependence modeling** via Student's t-copula (ν=5), grounded in the dependence frameworks of Demarta and McNeil (2005), to model fat tails and extreme co-movements across 20 Principal Risk Types (PRTs) and 100 Key Risk Indicators (KRIs).

3. **Econometric baselines** consisting of:
   - VAR with Diebold-Yilmaz variance decomposition (FEVD) connectedness: transparent and widely adopted for measuring spillover indices
   - CoVaR (Adrian & Brunnermeier): directly targeting systemic tail risk through conditional value-at-risk

4. **Network propagation methods** including:
   - DebtRank (Battiston et al., 2016): grounded in Eisenberg-Noe clearing payment models, propagating distress through cascading liabilities

5. **Advanced machine learning** via Graph Neural Networks (GNNs): learning non-linear, higher-order dependencies through message-passing architectures.

By fixing the ground truth network structure and varying the difficulty of the estimation problem (data volume, noise, tail dependence), this study produces empirical evidence directly transferable to practitioners: a financial institution facing 100 KRI series and limited monthly observations will know which technique to trust and what accuracy to expect.

## Competitor Analysis

Our methodological framework is evaluated against four established approaches in financial risk management:

### Diebold-Yilmaz Connectedness Framework (Academic Baseline)

Uses generalized forecast-error variance decompositions from vector autoregressions (VAR) to produce spillover indices.

**Strengths:**
- Transparent and theoretically grounded in VAR variance decomposition
- Widely adopted across academic and practitioner communities
- Produces stable estimates with limited data
- Naturally handles multiple risk types simultaneously

**Weaknesses:**
- Linear framework capturing only average co-movements, not tail dependence
- Cannot distinguish lead/lag relationships without careful ordering
- FEVD estimates sensitive to lag-length selection

**Reference:** Diebold & Yilmaz (2014), *Journal of Econometrics*, 182(1): 119-134

---

### Adrian & Brunnermeier CoVaR Framework (Academic Baseline)

Focuses on conditional value-at-risk to measure tail-risk spillovers from one risk type to another.

**Strengths:**
- Directly targets systemic tail risk, the most damaging contagion scenario
- Captures conditional distress linkages (i.e., "What is the impact of an extreme event in PRT X on PRT Y?")
- Grounded in risk management practice (VaR/CoVaR are standard risk metrics)

**Weaknesses:**
- Primarily pairwise rather than modeling full multi-node enterprise risk networks
- No enforcement of consistency constraints across simultaneous edge estimates
- Computationally intensive for large networks

**Reference:** Adrian & Brunnermeier (2016), *American Economic Review*, 106(7): 1705-1741

---

### Battiston et al. DebtRank (Propagation Benchmark)

Grounded in the Eisenberg–Noe clearing-payment model, it propagates distress through cascading liabilities.

**Strengths:**
- Direct measure of systemic loss through network propagation
- Models realistic contagion dynamics (shock spreads iteratively through network)
- Captures second-order and higher-order contagion effects

**Weaknesses:**
- Requires a known, granular exposure matrix (not a recovery method per se)
- Computationally expensive for real-time stress testing
- Sensitive to network structure assumptions

**Reference:** Battiston et al. (2016), *Statistics and Risk Modeling*, 33(3-4): 117-138

---

### Commercial GRC Platforms (Industry Solutions)

Enterprise software platforms (e.g., Moody's RiskCalc, SAS Risk Management, Kyriba, Numerix) provide static risk reporting, RAG threshold tracking, and compliance logging.

**Strengths:**
- Integrated enterprise risk architecture spanning multiple risk domains
- Regulatory compliance built-in
- Mature support and documentation

**Weaknesses:**
- Proprietary black-box logic unsuited for algorithmic recovery benchmarking
- High cost and vendor lock-in
- Limited flexibility for academic research or methodological innovation
- Do not model interconnectedness; treat risks as independent for dashboard display

---

## SWOT Analysis of This Study

### Strengths

- Explicitly models tail dependence via Student's t-copulas (ν=5), a characteristic feature of financial contagion
- Tests estimator recovery limits under strict data constraints relevant to African banking environments
- Provides rigorous benchmark comparisons across econometric and machine-learning models under controlled conditions
- Methods-study design breaks the circularity problem by planting ground truth and measuring recovery error
- Fully reproducible with open-source code and no proprietary data dependencies
- Comprehensive evaluation across six metrics (precision, recall, F1, AUC-ROC, edge weight correlation, impulse-response MSE)

### Weaknesses

- Relies on controlled synthetic KRI panel data rather than proprietary commercial bank data (intentional by design to eliminate circularity and enable reproducibility)
- Limited to 20 PRTs and 100 KRIs; results may not scale to institutions with more granular risk taxonomies
- Assumes fixed PRT structure; does not explore dynamic changes to risk taxonomy
- GNN implementation is academic proof-of-concept, not production-hardened
- Does not model structural breaks or regime changes in financial systems

### Opportunities

- Framework can be adopted by regional bank risk teams to stress-test their internal KRI reporting structures
- Can validate network recovery algorithms before deployment in live risk systems
- Methodology transferable to other domains (supply chain contagion, regulatory network analysis, etc.)
- Results can inform central bank and financial regulator risk monitoring frameworks
- Open-source release enables community contributions and extensions

### Threats

- Potential overfitting of advanced machine learning models (GNNs) when trained on noisy, short-horizon time series (main methodological risk)
- Synthetic data may not capture all features of real financial risk distributions
- Hyperparameter sensitivity of GNN may require extensive tuning for practitioners without ML expertise
- Computational cost of full benchmarking study (9,000 Monte Carlo experiments) limits accessibility for researchers without GPU resources

---

## Bibliography

Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015). Systemic risk and stability in financial networks. *American Economic Review*, 105(2), 564–608.

Acharya, V. V., & Skeie, O. B. (2011). A model of liquidity hoarding and term premia in inter-bank markets. *Journal of Monetary Economics*, 58(5), 436–447.

Acharya, V. V., Pedersen, L. H., Philippon, T., & Richardson, M. (2017). Measuring systemic risk. *Review of Financial Studies*, 30(1), 2–47.

Adrian, T., & Brunnermeier, M. K. (2016). CoVaR. *American Economic Review*, 106(7), 1705–1741.

Allen, F., & Gale, D. (2000). Financial contagion. *Journal of Political Economy*, 108(1), 1–33.

Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., & Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv preprint arXiv:1806.01261*.

Battiston, S., Caldarelli, G., D'Errico, M., & Gurciullo, S. (2016). Leveraging the network: A stress-test framework based on DebtRank. *Statistics and Risk Modeling*, 33(3–4), 117–138.

Beck, T., Cull, R., & Fuchs, M. (2018). Banking in Africa. In *Oxford Handbook of Banking* (3rd ed.). Oxford University Press.

Bello, Z. Y., Guo, K., & Newaz, K. M. S. (2022). Time-varying dynamics of African stock market integration. *Emerging Markets Review*, 52, 100917.

Boako, G., & Alagidede, P. (2018). African stock market integration. *Research in International Business and Finance*, 45, 370–382.

Brunnermeier, M. K. (2009). Deciphering the liquidity and credit crunch 2007–2008. *Journal of Economic Perspectives*, 23(1), 77–100.

Brunnermeier, M. K., & Pedersen, L. H. (2009). Market liquidity and funding liquidity. *Review of Financial Studies*, 22(6), 2201–2238.

Davies, J., Finlay, M., McLenaghen, T., & Wilson, D. (2006). Key risk indicators – Their role in operational risk management and measurement. *RiskBusiness International*.

Demarta, S., & McNeil, A. J. (2005). The t copula and related copulas. *International Statistical Review*, 73(1), 111–129.

Diamond, D. W., & Dybvig, P. H. (1983). Bank runs, deposit insurance, and liquidity. *Journal of Political Economy*, 91(3), 401–419.

Diebold, F. X., & Yilmaz, K. (2014). On the network topology of variance decompositions: Measuring the connectedness of financial firms. *Journal of Econometrics*, 182(1), 119–134.

Eisenberg, L., & Noe, T. H. (2001). Systemic risk in financial systems. *Management Science*, 47(2), 236–249.

Embrechts, P., McNeil, A., & Straumann, D. (2002). Correlation and dependence in risk management: Properties and pitfalls. In *Risk Management: Value at Risk and Beyond* (pp. 176–223). Cambridge University Press.

Freixas, X., Parigi, B., & Rochet, J. C. (2000). Systemic risk, interbank relations, and liquidity provision by the central bank. *Journal of Money, Credit and Banking*, 32(3), 611–638.

Gogas, P., Papadimitriou, T., & Karagkiozis, I. (2021). Machine learning in financial risk management: A systematic review. *Journal of Risk and Financial Management*, 14(5), 210.

Gorton, G., & Metrick, A. (2012). Securitized banking and the run on repo. *Journal of Financial Economics*, 104(3), 425–451.

Greenwood, R., Landier, A., & Thesmar, D. (2015). Aggressive corporate tax avoidance as a response to rising top marginal tax rates. *Journal of Finance*, 70(1), 123–150.

He, Z., & Xiong, W. (2012). Debt financing in asset markets. *American Economic Review*, 102(3), 88–94.

Imbierowicz, B., & Rauch, C. (2014). The relationship between liquidity risk and credit risk in banks. *Journal of Banking & Finance*, 40, 242–256.

Kipf, T., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.

McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative risk management: Concepts, techniques and tools* (2nd ed.). Princeton University Press.

---

*Literature review updated October 2026. Complete bibliography addressing all major sources cited in the project.*
