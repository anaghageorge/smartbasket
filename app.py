import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="SmartBasket",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SmartBasket")
st.subheader("Market Basket Analysis Using the Apriori Algorithm")

st.write(
    "Discover which products customers frequently purchase together "
    "using Data Mining."
)

st.divider()

# Upload dataset
st.header("📂 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Upload your transaction CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read line-by-line safely to handle variable items per transaction row
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.splitlines()

    transactions = []
    for line in lines:
        if line.strip():
            items = [item.strip() for item in line.split(",") if item.strip()]
            if items:
                transactions.append(items)

    # Create a preview dataframe
    data = pd.DataFrame([", ".join(t) for t in transactions], columns=["Transactions"])

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(data, use_container_width=True)

    # Remove empty transactions
    transactions = [t for t in transactions if t]

    # Find unique products
    all_products = sorted({
        product
        for transaction in transactions
        for product in transaction
    })

    if not transactions or not all_products:
        st.error("The uploaded dataset contains no valid products.")
        st.stop()

    # Create basket matrix
    basket = pd.DataFrame(
        False,
        index=range(len(transactions)),
        columns=all_products
    )

    for index, transaction in enumerate(transactions):
        for product in transaction:
            basket.loc[index, product] = True

    st.divider()

    # Dashboard statistics
    st.header("📊 Dataset Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(transactions))
    col2.metric("Unique Products", len(all_products))
    col3.metric("Total Product Entries", int(basket.sum().sum()))

    # Popular products chart
    st.subheader("🏆 Most Frequently Purchased Products")

    product_counts = basket.sum().sort_values(ascending=False)

    fig, ax = plt.subplots()

    product_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Products")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Product Purchase Frequency")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Apriori settings
    st.header("⚙️ Apriori Settings")

    col1, col2 = st.columns(2)

    with col1:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.1
        )

    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    # Use session state so the analysis doesn't disappear when selecting products
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False

    if st.button("🔍 Analyse Shopping Patterns"):
        st.session_state.analyzed = True

    if st.session_state.analyzed:

        frequent_items = apriori(
            basket,
            min_support=min_support,
            use_colnames=True
        )

        st.header("📊 Frequently Purchased Products")

        if frequent_items.empty:

            st.warning(
                "No frequent itemsets found. Try lowering Minimum Support."
            )

        else:

            frequent_items["Products"] = frequent_items[
                "itemsets"
            ].apply(lambda items: ", ".join(sorted(items)))

            frequent_display = frequent_items[
                ["Products", "support"]
            ].rename(columns={"support": "Support"})

            frequent_display["Support"] = (
                frequent_display["Support"].round(3)
            )

            st.dataframe(
                frequent_display,
                use_container_width=True
            )

            st.metric(
                "Frequent Itemsets Found",
                len(frequent_items)
            )

            st.download_button(
                "📥 Download Frequent Itemsets",
                data=frequent_display.to_csv(index=False),
                file_name="frequent_itemsets.csv",
                mime="text/csv"
            )

            # Association rules
            st.header("💡 Product Association Rules")

            rules = association_rules(
                frequent_items,
                metric="confidence",
                min_threshold=min_confidence
            )

            if rules.empty:

                st.info(
                    "No association rules found. "
                    "Try lowering Minimum Confidence."
                )

            else:

                rules["If Customer Buys"] = rules[
                    "antecedents"
                ].apply(lambda x: ", ".join(sorted(x)))

                rules["They May Also Buy"] = rules[
                    "consequents"
                ].apply(lambda x: ", ".join(sorted(x)))

                rules_display = rules[
                    [
                        "If Customer Buys",
                        "They May Also Buy",
                        "support",
                        "confidence",
                        "lift"
                    ]
                ].rename(
                    columns={
                        "support": "Support",
                        "confidence": "Confidence",
                        "lift": "Lift"
                    }
                )

                for column in ["Support", "Confidence", "Lift"]:
                    rules_display[column] = rules_display[
                        column
                    ].round(3)

                rules_display = rules_display.sort_values(
                    "Confidence",
                    ascending=False
                )

                st.dataframe(
                    rules_display,
                    use_container_width=True
                )

                st.metric(
                    "Association Rules Found",
                    len(rules_display)
                )

                st.download_button(
                    "📥 Download Association Rules",
                    data=rules_display.to_csv(index=False),
                    file_name="association_rules.csv",
                    mime="text/csv"
                )

                # Recommendations
                st.header("🛍️ Product Recommendations")

                selected_product = st.selectbox(
                    "Select a product",
                    all_products
                )

                recommendations = rules[
                    rules["antecedents"].apply(
                        lambda x: selected_product in x
                    )
                ].copy()

                if recommendations.empty:

                    st.info(
                        "No recommendations found for this product. "
                        "Try another product or lower the thresholds."
                    )

                else:

                    recommendations["consequents_str"] = recommendations["consequents"].apply(
                        lambda x: ", ".join(sorted(x))
                    )

                    unique_recommendations = recommendations.drop_duplicates(
                        subset=["consequents_str"]
                    ).sort_values("confidence", ascending=False)

                    for _, rule in unique_recommendations.head(5).iterrows():
                        products = rule["consequents_str"]
                        confidence_pct = rule["confidence"] * 100

                        st.success(
                            f"If a customer buys **{selected_product}**, "
                            f"recommend **{products}** "
                            f"(Confidence: {confidence_pct:.1f}%)"
                        )

                st.success("Analysis completed successfully!")

else:
    st.info("Upload a CSV file to begin your analysis.")

st.divider()

st.caption(
    "SmartBasket | Data Mining Project | Apriori Algorithm"
)