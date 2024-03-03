import 'package:flutter/material.dart';

import '../widget/customSearchBar.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromRGBO(43, 62, 72, 1),
      body: Center(
        child: Column(
          children: [
            SizedBox(
              height: 400,
            ),
            Container(
              child: Column(
                children: [
                  Text(
                    "Google",
                    style: TextStyle(
                      fontSize: 80,
                      color: Color.fromRGBO(132, 209, 244, 1),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(
                    height: 30,
                  ),
                  CustomSearchBar(
                    barWidth: 600,
                    barHeight: 50,
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
