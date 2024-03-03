import 'package:flutter/material.dart';
import 'package:search_ui/widget/customSearchBar.dart';

class SearchResult extends StatelessWidget {
  const SearchResult({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Row(
            children: [
              Text(
                "Google",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: Color.fromRGBO(43, 62, 72, 1),
                ),
              ),
              SizedBox(
                width: 20,
              ),
              CustomSearchBar(barWidth: 300, barHeight: 25)
            ],
          )
        ],
      ),
    );
  }
}
