# Data

Raw data is not committed and was not downloaded to this machine.

- **Kaggle slug:** `olistbr/brazilian-ecommerce`
- **License (per Kaggle metadata):** CC BY-NC-SA 4.0. Non-commercial use only.
- **Read path:** `/kaggle/input/brazilian-ecommerce/`. The data is read in place from a Kaggle Notebook with the dataset attached. It is never copied into this repository.
- **Metadata read date:** 2026-11-08 (plan Appendix A, Kaggle dataset metadata only).
- **Files read on this machine:** none.

## Files used

- `olist_orders_dataset.csv`: `order_id`, `customer_id`, `order_status`, `order_purchase_timestamp`
- `olist_order_items_dataset.csv`: `order_id`, `product_id`, `price`, `freight_value`
- `olist_customers_dataset.csv`: `customer_id`, `customer_unique_id`

Customers are identified by `customer_unique_id`, because `customer_id` is one ID per order in Olist. Confirm the file names and columns on the Kaggle page before the first Kaggle run.
