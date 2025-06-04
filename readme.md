curl -X 'POST' \
  'http://localhost:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_message": "I am going to Paris, what should I see?"
}'

curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_message": "What are the latest credit card no-cost EMI offers in UK banks?"
}'


curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_message": "What are the latest credit card no-cost EMI offers in UK banks?"
}'


{
  "tenantName": "Credit card",
  "tenantDetails": "Barclays bank UK Credit card department, handling credit card operations and offers.",
  "offerForm": [
    "principalAmount",
    "tenure",
    "EMI",
    "interestRate"
  ],
  "offerType": "no cost EMI",
  "existingOffers": [],
  "metadata": "Requires the best competitive offer."
}


curl -X 'POST' \
  'http://127.0.0.1:8080/chat' \
  -H 'Content-Type: application/json' \
  -d '{
    "tenantName": "Credit card",
    "tenantDetails": "Barclays bank UK Credit card department, handling credit card operations and offers.",
    "offerForm": ["principalAmount", "tenure", "EMI", "interestRate"],
    "offerType": "no cost EMI",
    "existingOffers": [],
    "metadata": "Requires the best competitive offer.",
    "userQuery": "What are the latest credit card no-cost EMI offers in UK banks?"
  }'






response will be 
{
  "tenantName": "Credit card",
  "tenantDetails": "Barclays bank UK Credit card department, handling credit card operations and offers.",
  "offerForm": { 
    "principalAmount":{genrated value}, 
    "tenure":{genrated value}, 
    "EMI":{genrated value}, 
    "interestRate":{genrated value}
  ],
  "offerType": "no cost EMI",
"comparisontable":{genrated value},
}
