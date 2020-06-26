# API Documentation

![codeclimate badge](https://api.codeclimate.com/v1/badges/6324d14f8c5711ef6d82/maintainability)

#### API deployed at [AWS Elastic Beanstalk](https://8rq6v9dni0.execute-api.us-east-1.amazonaws.com/) <br>

## Getting started

To get the server running locally:

- Clone this repo
- **pipenv install** to install all required dependencies
- **pipenv shell** to activate the virtual environment
- **FLASK_APP=application.py** to set the flask app environment variable
- **flask run** to start the flask server.

### Backend framework goes here

🚫 Why did you choose this framework?

-    Point One
-    Point Two
-    Point Three
-    Point Four

## Endpoints

#### Routes

| Method | Endpoint                 | Parameters                      | Description                                             |
| ------ | ------------------------ | ------------------------------- | ------------------------------------------------------- |
| POST   | `/sentiment/`            | text                            | Returns predicted sentiment of a review.                |
| POST   | `/summarization/`        | text                            | Returns a summary of a review.                          |
| GET    | `/business_info/`        | city, name, address, categories | List businesses the recommender recognizes.             |
| GET    | `/infer_recommendations/`| business_id, stars              | Generate recommendations from business_ids and ratings. |



# Data Model

#### Businesses

---

```
{
  business_id: UUID
  name: STRING
  city: STRING
  address: BOOLEAN
  categories: STRING
}
```

#### USERS

---

```
{
  id: UUID
  organization_id: UUID foreign key in ORGANIZATIONS table
  first_name: STRING
  last_name: STRING
  role: STRING [ 'owner', 'supervisor', 'employee' ]
  email: STRING
  phone: STRING
  cal_visit: BOOLEAN
  emp_visit: BOOLEAN
  emailpref: BOOLEAN
  phonepref: BOOLEAN
}
```

## 2️⃣ Actions

🚫 This is an example, replace this with the actions that pertain to your backend

`getOrgs()` -> Returns all organizations

`getOrg(orgId)` -> Returns a single organization by ID

`addOrg(org)` -> Returns the created org

`updateOrg(orgId)` -> Update an organization by ID

`deleteOrg(orgId)` -> Delete an organization by ID
<br>
<br>
<br>
`getUsers(orgId)` -> if no param all users

`getUser(userId)` -> Returns a single user by user ID

`addUser(user object)` --> Creates a new user and returns that user. Also creates 7 availabilities defaulted to hours of operation for their organization.

`updateUser(userId, changes object)` -> Updates a single user by ID.

`deleteUser(userId)` -> deletes everything dependent on the user

## Environment Variables

In order for the app to function correctly, the user must set up their own environment variables.

create a .env file that includes the following:

    
    *  ACCESS_KEY_ID - access key id for an S3 bucket containing pickled lightFM Dataset and Recommender objects.
    *  SECRET_ACCESS_KEY - secret access key for an S3 bucket containing pickled lightFM Dataset and Recommender objects.
    *  AWS_RDS_HOST - endpoint for a Postgres DB on AWS RDS
    *  AWS_RDS_PORT - port for a Postgres DB on AWS RDS
    *  AWS_RDS_USER - user for a Postgres DB on AWS RDS
    *  AWS_RDS_PASS - password for a Postgres DB on AWS RDS
    *  AWS_RDS_DB - database name for a Postgres DB on AWS RDS
    
    
## Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a [code of conduct](./code_of_conduct.md). Please follow it in all your interactions with the project.

### Issue/Bug Request

 **If you are having an issue with the existing project code, please submit a bug report under the following guidelines:**
 - Check first to see if your issue has already been reported.
 - Check to see if the issue has recently been fixed by attempting to reproduce the issue using the latest master branch in the repository.
 - Create a live example of the problem.
 - Submit a detailed bug report including your environment & browser, steps to reproduce the issue, actual and expected outcomes,  where you believe the issue is originating from, and any potential solutions you have considered.

### Feature Requests

We would love to hear from you about new features which would improve this app and further the aims of our project. Please provide as much detail and information as possible to show us why you think your new feature should be implemented.

### Pull Requests

If you have developed a patch, bug fix, or new feature that would improve this app, please submit a pull request. It is best to communicate your ideas with the developers first before investing a great deal of time into a pull request to ensure that it will mesh smoothly with the project.

Remember that this project is licensed under the MIT license, and by submitting a pull request, you agree that your work will be, too.

#### Pull Request Guidelines

- Ensure any install or build dependencies are removed before the end of the layer when doing a build.
- Update the README.md with details of changes to the interface, including new plist variables, exposed ports, useful file locations and container parameters.
- Ensure that your code conforms to our existing code conventions and test coverage.
- Include the relevant issue number, if applicable.
- You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

### Attribution

These contribution guidelines have been adapted from [this good-Contributing.md-template](https://gist.github.com/PurpleBooth/b24679402957c63ec426).

## Documentation

See [Frontend Documentation](🚫link to your frontend readme here) for details on the fronend of our project.
🚫 Add DS iOS and/or Andriod links here if applicable.
