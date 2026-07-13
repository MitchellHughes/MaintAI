import { useEffect, useState } from "react";
import { getMetrics } from "../services/api";

export default function MetricsTable() {

    const [metrics, setMetrics] = useState([]);
    const [summary, setSummary] = useState(null);

    useEffect(() => {

        async function loadMetrics() {

            try {

                const data =
                    await getMetrics();

                console.log(data);

                // separate summary rows
                const summaryRows =
                    data.filter(
                        row =>
                            row["Unnamed: 0"] === "accuracy"
                            ||
                            row["Unnamed: 0"] === "macro avg"
                    );

                const classRows =
                    data.filter(
                        row =>
                            ![
                                "accuracy",
                                "macro avg",
                                "weighted avg"
                            ].includes(
                                row["Unnamed: 0"]
                            )
                    );

                setMetrics(classRows);

                if (
                    summaryRows.length
                ) {

                    setSummary(
                        summaryRows[0]
                    );

                }

            }

            catch (err) {

                console.error(
                    "Metrics load failed",
                    err
                );

            }

        }

        loadMetrics();

    }, []);

    return (

        <div
            style={{
                marginBottom: "30px"
            }}
        >

            <h2>
                Model Metrics
            </h2>

            {

            summary && (

                <div
                    style={{
                        padding: "10px",
                        border: "1px solid #dddddd",
                        borderRadius: "8px",
                        marginBottom: "20px"
                    }}
                >

                    <h3>
                        Overall Summary
                    </h3>

                    <p>
                        Precision:
                        {" "}
                        {
                        Number(
                            summary.precision
                        ).toFixed(2)
                        }
                    </p>

                    <p>
                        Recall:
                        {" "}
                        {
                        Number(
                            summary.recall
                        ).toFixed(2)
                        }
                    </p>

                    <p>
                        F1:
                        {" "}
                        {
                        Number(
                            summary["f1-score"]
                        ).toFixed(2)
                        }
                    </p>

                </div>

            )

            }

            <table>

                <thead>

                    <tr>

                        <th>Class</th>

                        <th>Precision</th>

                        <th>Recall</th>

                        <th>F1</th>

                        <th>Support</th>

                    </tr>

                </thead>

                <tbody>

                {

                metrics.length

                ?

                metrics

                .sort(

                    (
                        a,
                        b
                    ) =>

                    b["f1-score"]

                    -

                    a["f1-score"]

                )

                .map(

                    (
                        row,
                        i
                    ) => (

                    <tr key={i}>

                        <td>

                            {
                            row["Unnamed: 0"]
                            }

                        </td>

                        <td>

                            {
                            Number(
                                row.precision
                            ).toFixed(2)
                            }

                        </td>

                        <td>

                            {
                            Number(
                                row.recall
                            ).toFixed(2)
                            }

                        </td>

                        <td

                        style={{

                        color:

                        row["f1-score"] >= 0.8

                        ?

                        "green"

                        :

                        row["f1-score"] >= 0.6

                        ?

                        "orange"

                        :

                        "red"

                        }}

                        >

                            {
                            Number(
                                row["f1-score"]
                            ).toFixed(2)
                            }

                        </td>

                        <td>

                            {
                            row.support
                            }

                        </td>

                    </tr>

                ))

                :

                <tr>

                    <td
                        colSpan="5"
                    >

                        Loading metrics...

                    </td>

                </tr>

                }

                </tbody>

            </table>

        </div>

    );

}
